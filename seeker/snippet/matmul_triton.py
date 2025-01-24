#date: 2025-01-24T16:46:51Z
#url: https://api.github.com/gists/4cf873c3354e2bbe8434b611c6bf0b11
#owner: https://api.github.com/users/keshavchand

import triton
import triton.language as tl
import torch
import torch.nn.functional as F

import math

@triton.jit
def _relu_triton(
		A,
		rowsA, colsA
	):
	return tl.where(A > 0, A, 0)
pass

@triton.jit
def _matmul_triton(
		A, B, C,
		rowsA, colsA,
		rowsB, colsB,
		rowsC, colsC,

		activationFnId,

		blockSizeRows: tl.constexpr, 
		blockSizeCols: tl.constexpr,
		blockSizeTile: tl.constexpr,
	):
	cRows = tl.program_id(0)	
	cCols = tl.program_id(1)	

	innerTiles = tl.cdiv(colsA, blockSizeTile)
	res = tl.zeros((blockSizeRows, blockSizeCols), dtype=tl.float32)
	for i in range(innerTiles):
		aRow = cRows * blockSizeRows + tl.arange(0, blockSizeRows)[:, None]
		aCol = i     * blockSizeTile + tl.arange(0, blockSizeTile)[None, :]

		bRow = i     * blockSizeTile + tl.arange(0, blockSizeTile)[:, None]
		bCol = cCols * blockSizeCols + tl.arange(0, blockSizeCols)[None, :]

		aMask = (aRow < rowsA) & (aCol < colsA)
		bMask = (bRow < rowsB) & (bCol < colsB)
		aData = tl.load(A + aRow * colsA + aCol, mask = aMask, other = 0.0)
		bData = tl.load(B + bRow * colsB + bCol, mask = bMask, other = 0.0)

		# without the input_precision="ieee" it returns an error 
		# IndexError: map::at
		# Same issue on google cloud too
		res += tl.dot(aData, bData, input_precision="ieee")
	pass

	if activationFnId == 1:
		res = _relu_triton(res, blockSizeRows, blockSizeCols)

	cRow = cRows * blockSizeRows + tl.arange(0, blockSizeRows)[:, None]
	cCol = cCols * blockSizeCols + tl.arange(0, blockSizeCols)[None, :]
	cMask = (cRow < rowsC) & (cCol < colsC)
	tl.store(C + cRow * colsC + cCol, res, mask=cMask)
pass

activationFns = {
	'relu': 1,
}

def matmul(A: torch.Tensor, B: torch.Tensor, activationFn: str = 'relu', blockSize = (16, 16), tileSize = 16,) -> torch.Tensor:	
	assert A.shape[1] == B.shape[0]
	assert A.device == B.device
	assert activationFn in activationFns 
	
	C = torch.empty((A.shape[0], B.shape[1]), dtype=A.dtype, device=A.device)
	grid = (math.ceil(C.shape[0] / blockSize[0]), math.ceil(C.shape[1] / blockSize[1]))
	_matmul_triton[grid](
		A, B, C,
		A.shape[0], A.shape[1],
		B.shape[0], B.shape[1],
		C.shape[0], C.shape[1],

		activationFns[activationFn],
		blockSize[0], blockSize[1], tileSize,
	)

	return C	
pass



rows, cols = 2048, 2048
A = torch.arange(rows * cols, dtype=torch.float32, device="cuda").reshape(rows, cols) / (rows * cols)
B = torch.arange(rows * cols, dtype=torch.float32, device="cuda").reshape(rows, cols) / (rows * cols)

ref_C = A @ B
zeroTensor = torch.zeros((1, ), device=A.device)
ref_C_relu = ref_C.maximum(zeroTensor)

triton_C_relu = matmul(A, B)

try:
	torch.testing.assert_close(ref_C_relu, triton_C_relu)
except Exception as e:
	print(ref_C_relu[:16, :16])
	print(triton_C_relu[:16, :16])
