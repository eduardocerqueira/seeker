#date: 2024-12-30T16:24:58Z
#url: https://api.github.com/gists/de59bdf506fe2a7db50d621aff968d56
#owner: https://api.github.com/users/natanaelgsneto

import cv2

# Carregar os modelos pré-treinados
proto_file = "colorization_deploy_v2.prototxt"
model_file = "colorization_release_v2.caffemodel"
points_file = "pts_in_hull.npy"

# Baixe os arquivos necessários:
# https://github.com/richzhang/colorization/tree/master/models

# Carregar os arquivos do modelo
net = cv2.dnn.readNetFromCaffe(proto_file, model_file)
points = np.load(points_file)

# Adicionar pontos ao modelo
pts = points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype("float32")]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Carregar a imagem em escala de cinza
image = cv2.imread("sua_imagem_pb.jpg")
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

# Redimensionar a camada L para o tamanho esperado
L = cv2.split(lab)[0]
L = cv2.resize(L, (224, 224))
L -= 50

# Prever a camada 'ab' para colorir a imagem
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

# Redimensionar a camada 'ab' para o tamanho original da imagem
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

# Combinar com a camada L original e converter de volta para RGB
colorized = np.concatenate((cv2.split(lab)[0][:, :, np.newaxis], ab), axis=2)
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = (colorized * 255).astype("uint8")

# Exibir e salvar a imagem colorida
cv2.imshow("Colorized", colorized)
cv2.imwrite("imagem_colorida.jpg", colorized)
cv2.waitKey(0)
cv2.destroyAllWindows()
