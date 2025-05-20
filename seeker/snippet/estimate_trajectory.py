#date: 2025-05-20T17:09:57Z
#url: https://api.github.com/gists/80a1f32e191ddb8271b923db35385a70
#owner: https://api.github.com/users/vagizD

from collections import defaultdict

from common.dataset import Dataset
from common.trajectory import Trajectory

import numpy as np
import pandas as pd
import cv2

import pickle
import os


def read(path):
    with open(path, 'rb') as fhandle:
        obj = pickle.load(fhandle)
    return obj


def save(obj, path):
    with open(path, 'wb') as fhandle:
        pickle.dump(obj, fhandle)


def quaternion_to_rotation_matrix(quaternion):
    """
    Generate rotation matrix 3x3 from the unit quaternion.
    Input:
    qQuaternion -- tuple consisting of (qx,qy,qz,qw) where
    (qx,qy,qz,qw) is the unit quaternion.
    7Output:
    matrix -- 3x3 rotation matrix
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    eps = np.finfo(float).eps * 4.0
    assert nq > eps
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3]),
        (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3]),
        (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1])
    ), dtype=np.float64)


def get_data(data_dir):
    rgb_list = Dataset.get_rgb_list_file(data_dir)
    images   = pd.read_csv(rgb_list, names=['#frame_id', 'img_path'], sep=' ').to_dict(orient='list')
    images  = dict(zip(images['#frame_id'], images['img_path']))
    images  = {k: os.path.join(data_dir, v) for k, v in images.items()}

    poses_path = Dataset.get_known_poses_file(data_dir)
    poses      = pd.read_csv(poses_path, sep=' ', index_col='#frame_id')

    intrinsics_path = Dataset.get_intrinsics_file(data_dir)
    intrinsics      = pd.read_csv(intrinsics_path, sep=' ').iloc[0, :].values[:-1]

    return images, poses, intrinsics



def get_descriptor_and_keypoints(img_path):
    img = cv2.imread(img_path, 0)
    orb = cv2.ORB_create()
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    return kp, des


def get_descriptors_and_keypoints(images, save_path=None):
    # des_path = os.path.join(save_path, 'descriptors.pickle')
    # kp_path  = os.path.join(save_path, 'keypoints.pickle')

    # if os.path.exists(des_path):
    #     descriptors = read(des_path)
    #     keypoints   = read(kp_path)
    #
    # else:
    descriptors, keypoints = {}, {}
    for image_id, image_path in images.items():
        kp, des = get_descriptor_and_keypoints(image_path)
        descriptors[image_id] = des
        keypoints[image_id]   = kp

        # if save_path is not None:
        #     os.mkdir(save_path)
        #     save(descriptors, des_path)
        #     save(keypoints, kp_path)

    return keypoints, descriptors


def get_inliers(support_images, keypoints, descriptors):
    inliers = {}
    si = list(support_images.keys())
    for i, id1 in enumerate(si):
        for id2 in si[i + 1:]:
            ip = get_inliers_pair(
                keypoints[id1], keypoints[id2],
                descriptors[id1], descriptors[id2]
            )
            if ip is not None:
                inliers[(id1, id2)] = ip
                # break
        # break
    return inliers


def get_inliers_pair(kp1, kp2, des1, des2):
    # FLANN_INDEX_LSH = 6
    # index_params = dict(algorithm=FLANN_INDEX_LSH,
    #                     table_number=12,
    #                     key_size=20,
    #                     multi_probe_level=2)
    # search_params = dict(checks=50)
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1, des2, k=2)
    # matches = [m for m in matches if len(m) == 2]

    # BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    inliers = []  # (queryIdx, trainIdx)
    pts1    = []
    pts2    = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):  # pt1 - best match, pt2 - second best match
        if m.distance < 0.75 * n.distance:
            inliers.append((m.queryIdx, m.trainIdx))
            # inliers.append((m.queryIdx, m.trainIdx, m.distance))
            # inliers.append((kp1[m.queryIdx], kp2[m.trainIdx], m.distance))
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    if len(inliers) < 8:
        return None

    # filter using RANSAC and fundamental matrix
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    if len(inliers) == 0:
        raise ValueError
    inliers = np.array(inliers)[mask.ravel() == 1]

    return inliers


def extract_inliers(img1, img2, inliers):
    if img1 < img2:
        return inliers.get((img1, img2), None)
    x = inliers.get((img2, img1), None)
    if x is not None:
        x = [(p[1], p[0]) for p in x]
    return x


def get_tracks(support_images, inliers):
    tracks = []  # ( (img1, img2, img3), (kp1, kp2, kp3) )
    si = list(support_images.keys())

    n = len(si)
    pool = [(x, y, z) for x in si for y in si[x:] for z in si[y:]]
    # idx = np.random.choice(np.arange(len(pool)), size=n * n * 5)
    # pool = [pool[i] for i in idx]
    for (i, j, k) in pool:
    # for i in range(len(support_images)):
    #     for j in range(len(support_images)):
    #         for k in range(len(support_images)):
                # k = j + 2

                if i == j or i == k or j == k:
                    continue

                img1 = si[i]
                img2 = si[j]
                img3 = si[k]

                inliers12 = extract_inliers(img1, img2, inliers)
                inliers23 = extract_inliers(img2, img3, inliers)
                inliers13 = extract_inliers(img1, img3, inliers)
                # inliers12 = inliers.get((img1, img2), None)
                # inliers23 = inliers.get((img2, img3), None)
                # inliers13 = inliers.get((img1, img3), None)
                if inliers12 is None or inliers23 is None or inliers13 is None:
                    continue

                # p1 -> p2 = pp1 -> pp2 = ppp1 -> ppp2 = p1
                for (p1, p2) in inliers12:
                    for (pp1, pp2) in inliers23:
                        if p2 != pp1:
                            continue
                        for (ppp2, ppp1) in inliers13:
                            if pp2 != ppp1 or ppp2 != p1:
                                continue
                            tracks.append(
                                ( (img1, img2, img3), (p1, pp1, ppp1) )
                            )

    return tracks


def get_intrinsics_matrix(intrinsics):
    f_x, f_y, c_x, c_y = intrinsics
    K = np.array([
        [f_x,   0, c_x],
        [  0, f_y, c_y],
        [  0,   0,   1]
    ])
    return K


def get_projection_matrix(img, poses, K):
    translation = poses.loc[img, :].values[:3].reshape((1, -1))
    quaternion  = poses.loc[img, :].values[3:]

    rotation = np.linalg.inv(quaternion_to_rotation_matrix(quaternion))
    rodrigues, _ = cv2.Rodrigues(rotation)
    translation  = -1.0 * np.matmul(rotation, translation.T)

    projection_matrix = np.matmul(
        K, np.concatenate((rotation, translation), axis=1)
    )
    return (projection_matrix, rodrigues, translation)


def get_triangulation(tracks, projection_matrices, keypoints, K):
    scene_points = {}  # track <--> 3D point
    image2scene = defaultdict(lambda: {})

    # for each track, solve OLS and get 3D point
    image2inliers = defaultdict(lambda: set())
    for track in tracks:
        ( (img1, img2, img3), (kp1, kp2, kp3) ) = track
        pm1 = projection_matrices[img1]  # (projection_matrix, rodrigues_matrix, translation)
        pm2 = projection_matrices[img2]
        pm3 = projection_matrices[img3]

        pt1 = keypoints[img1][kp1]  # cv2.KeyPoint
        pt2 = keypoints[img2][kp2]
        pt3 = keypoints[img3][kp3]

        # build matrix equation
        n = 3
        A = np.zeros((2 * n, 4))
        points = [pt1, pt2, pt3]
        pms    = [pm1, pm2, pm3]

        for i in range(n):
            pt = points[i].pt
            pm = pms[i][0]

            u, v = pt
            A[    2 * i, :] = u * pm[2, :] - pm[0, :]
            A[2 * i + 1, :] = v * pm[2, :] - pm[1, :]

        # convert AX = 0 to OLS task
        A13 = A[:, :3]
        A4  = A[:, 3:4]
        pt_3D = - np.linalg.inv(A13.T @ A13) @ A13.T @ A4  # X, 3 coords
        # pt_4D = np.concatenate((pt_3D, [[1]]))
        # pt_3D = pt_3D.flatten() / pt_3D[3]  # now in homogeneous coordinates

        # reproject back to images
        is_good = True
        for i in range(n):
            pt = points[i].pt
            rodrigues = pms[i][1]
            translation = pms[i][2]
            reprojection, _ = cv2.projectPoints(
                pt_3D[0:3].T,
                rodrigues,
                translation,
                K,
                None
            )
            error = np.linalg.norm(pt - reprojection)
            if error > 7:
                is_good = False
                break

        if is_good:
            image2inliers[img1].add(kp1)
            image2inliers[img2].add(kp2)
            image2inliers[img3].add(kp3)

            image2scene[img1][np.array(pt1.pt).tobytes()] = pt_3D
            image2scene[img2][np.array(pt2.pt).tobytes()] = pt_3D
            image2scene[img3][np.array(pt3.pt).tobytes()] = pt_3D

            scene_points[track] = pt_3D

    return scene_points, image2inliers, image2scene


def get_inliers2(support_images, unknown_images, keypoints_filt, descriptors_filt):
    inliers = {}
    si = list(support_images.keys())
    ui = list(unknown_images.keys())

    for id1 in si:
        for id2 in ui:
            ip = get_inliers_pair(
                keypoints_filt[id1], keypoints_filt[id2],
                descriptors_filt[id1], descriptors_filt[id2]
            )
            if ip is not None:
                inliers[(id1, id2)] = ip
                # break
        # break
    return inliers


def get_poses(unknown_images, image2scene, inliers2, keypoints_filt, K):
    poses = {}

    for img in unknown_images:
        img_inliers = [k for k in inliers2 if k[1] == img]  # get inliers for unknown image

        scene_pts   = []
        unknown_pts = []
        for pair in img_inliers:
            inliers = inliers2[pair]
            support_img = pair[0]
            support_kp_all  = keypoints_filt[support_img]
            support_kp_idx  = [inlier[0] for inlier in inliers]
            support_kp      = [support_kp_all[i] for i in support_kp_idx]

            unknown_kp_all  = keypoints_filt[img]
            unknown_kp_idx  = [inlier[1] for inlier in inliers]
            unknown_kp      = [unknown_kp_all[i] for i in unknown_kp_idx]

            scene_pt   = [image2scene[support_img][np.array(pt.pt).tobytes()] for pt in support_kp]
            unknown_pt = [np.array(pt.pt) for pt in unknown_kp]
            scene_pts.extend(scene_pt)
            unknown_pts.extend(unknown_pt)

        objectPoints = np.array(scene_pts)
        imagePoints  = np.array(unknown_pts)

        try:
            _, rvec, tvec, _ = cv2.solvePnPRansac(objectPoints, imagePoints, K, None)
        except Exception as e:
            print(objectPoints.shape)
            print(imagePoints.shape)
            raise e

        rotation, _ = cv2.Rodrigues(rvec)
        rotation = np.linalg.inv(rotation)

        translation = -1.0 * np.matmul(rotation, tvec.reshape(1, -1).T)

        tm = np.zeros((4, 4), dtype=np.float32)
        tm[:3, :3] = rotation
        tm[:3, 3] = translation.ravel()
        tm[3, 3] = 1

        poses[img] = tm
    return poses


def estimate_trajectory(data_dir, out_dir):
    if not os.environ.get('CHECKER'):
        save_path = os.path.join(out_dir, 'save_results')
    else:
        save_path = None

    images, poses, intrinsics = get_data(data_dir)

    support_images = {k: v for k, v in images.items() if 'with_poses' in v}  # sorted
    unknown_images = {k: v for k, v in images.items() if k not in support_images}
    assert sorted(list(support_images.keys())) == list(support_images.keys())

    K = get_intrinsics_matrix(intrinsics)
    projection_matrices = {
        img: get_projection_matrix(img, poses, K)
        for img in support_images
    }

    keypoints, descriptors = get_descriptors_and_keypoints(images, save_path=save_path)
    inliers = get_inliers(support_images, keypoints, descriptors)
    tracks = get_tracks(support_images, inliers)
    scene_points, image2inliers, image2scene = get_triangulation(tracks, projection_matrices, keypoints, K)

    keypoints_filt, descriptors_filt = {}, {}
    for img in support_images:
        ids = image2inliers[img]
        all_ids = list(range(len(keypoints[img])))
        keypoints_filt[img]   = [keypoints[img][i] for i in all_ids if i in ids]
        descriptors_filt[img] = np.array([descriptors[img][i] for i in all_ids if i in ids])
    for img in unknown_images:
        keypoints_filt[img]   = keypoints[img]
        descriptors_filt[img] = descriptors[img]

    inliers2 = get_inliers2(
        support_images, unknown_images, keypoints_filt, descriptors_filt
    )
    unknown_poses = get_poses(unknown_images, image2scene, inliers2, keypoints_filt, K)

    m = len(images)
    n = len(support_images)
    print("\n\n")
    print(f"# of images              : {m}")
    print(f"# of support images      : {n}")
    print(f"Keypoints per image      : {np.mean([len(k) for k in keypoints.values()]):.2f}")
    print(f"# of support pairs       : {n * (n - 1) // 2}")
    print(f"# of pairs with inliers  : {len(inliers)}")
    print(f"Inliers per support pair : {np.mean([len(t) for t in inliers.values()]):.2f}")
    print(f"Tracks found             : {len(tracks)}")
    print(f"Tracks per inliers pair  : {len(tracks) / len(inliers):.2f}")
    print(f"Scene points found       : {len(scene_points)}")
    print(f"Scene points per track   : {len(scene_points) / len(tracks):.2f}")
    print(f"# of pairs with inliers2 : {len(inliers2)}")
    print(f"Inliers2 per support pair: {np.mean([len(t) for t in inliers2.values()]):.2f}")

    trajectory = {}
    for img in poses.index:
        trajectory[img] = Trajectory.to_matrix4(poses.loc[img].values.ravel())
    for img in unknown_poses:
        trajectory[img] = unknown_poses[img]

    Trajectory.write(Dataset.get_result_poses_file(out_dir), trajectory)