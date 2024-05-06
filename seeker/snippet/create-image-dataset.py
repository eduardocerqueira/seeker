#date: 2024-05-06T16:59:38Z
#url: https://api.github.com/gists/895f07e7ce0cb85e38fe2d00249dc8f3
#owner: https://api.github.com/users/gabriel-ab

import cv2
from pathlib import Path

Q, SPACE, *NUMS = map(ord, 'q 1234567890')

def create_dataset(savedir: str | Path, classes: list[str]):
    savedir = Path(savedir)
    for c in classes:
        (savedir / c).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    selected = 0
    current = [0] * len(classes)
    while ok:
        cv2.imshow('Dataset Creator', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == Q:
            break
        elif key in NUMS:
            try:
                selected = NUMS.index(key)
                print(f'Class "{classes[selected]}" selected')
            except IndexError:
                print(f'Invalid Option, select numbers from 1 to {len(classes)}')
        elif key == SPACE:
            filepath = savedir / classes[selected] / f'{current[selected]:04d}.jpg'
            current[selected] += 1
            cv2.imwrite(str(filepath), frame)
            print('Annotated:', current, 'Saved file:', filepath)

        ok, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print('Press Q to exit')
    create_dataset(
        savedir = 'data',
        classes = ['class1', 'class2', ...]
    )
