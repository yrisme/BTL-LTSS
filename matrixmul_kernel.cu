/*Phép nhân ma trận: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"
#define TILE_WIDTH 32 

//kernel nhân ma trận
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
	// Khai báo bộ nhớ chia sẻ (shared memory) cho các khối (tile) của M và N
	__shared__ float tileMs[TILE_WIDTH][TILE_WIDTH];
	__shared__ float tileNs[TILE_WIDTH][TILE_WIDTH];

	// Lấy chỉ số của luồng trong block (threadIdx) và chỉ số của block trong grid (blockIdx)
	int tx = threadIdx.x; int ty = threadIdx.y;
	int bx = blockIdx.x; int by = blockIdx.y;

	// tọa độ của phần tử đích (trong ma trận kết quả P)
	int row = by * TILE_WIDTH + ty;    // Hàng của phần tử đích
	int column = bx * TILE_WIDTH + tx; // Cột của phần tử đích

	float pValue = 0; // Biến lưu trữ giá trị của phần tử P[row][column] đang được tính toán

	// tính toán giá trị của phần tử đích
	// Vòng lặp này duyệt qua các khối (tile) theo chiều rộng của M (hoặc chiều cao của N)
	// để tính toán tích vô hướng hoàn chỉnh cho P[row][column]
	for(int i=0;i<ceilf(M.width/(float)TILE_WIDTH);i++){
		// di chuyển các khối (tile) và cập nhật giá trị bộ nhớ chia sẻ cho các vị trí khối mới
		
		// Tải một phần tử từ ma trận M (global memory) vào khối tileMs (shared memory)
		// Kiểm tra biên để đảm bảo không truy cập ngoài vùng nhớ của M
		if(row < M.height && (i*TILE_WIDTH + tx)<M.width)
			tileMs[ty][tx] = M.elements[row*M.width + i*TILE_WIDTH + tx];
		else
			tileMs[ty][tx] = 0; // Nếu ngoài biên, gán giá trị 0

		// Tải một phần tử từ ma trận N (global memory) vào khối tileNs (shared memory)
		// Kiểm tra biên để đảm bảo không truy cập ngoài vùng nhớ của N
		if(column < N.width && (i*TILE_WIDTH + ty)<N.height)
			tileNs[ty][tx] = N.elements[(i*TILE_WIDTH + ty)*N.width + column];
		else
			tileNs[ty][tx] = 0; // Nếu ngoài biên, gán giá trị 0

		// sau khi toàn bộ giá trị của khối (tile) đã có sẵn (trong shared memory), tiếp tục
		__syncthreads(); // Đồng bộ hóa tất cả các luồng trong block.
		                 // Đảm bảo tất cả các luồng đã tải xong dữ liệu vào tileMs và tileNs
		                 // trước khi bắt đầu tính toán.

		// Thực hiện phép nhân giữa các phần tử của tileMs và tileNs (đang nằm trong shared memory)
		// để cộng dồn vào pValue
		for(int j=0;j<TILE_WIDTH;j++)
			pValue += tileMs[ty][j] * tileNs[j][tx];
		
		// sau khi toàn bộ giá trị của khối (tile) đã được sử dụng, tiếp tục
		__syncthreads(); // Đồng bộ hóa tất cả các luồng trong block.
		                 // Đảm bảo tất cả các luồng đã hoàn thành việc sử dụng dữ liệu
		                 // từ tileMs và tileNs của "phase" hiện tại trước khi chuyển sang "phase" tiếp theo.
	}
	// boundary check // kiểm tra biên
	// Sau khi tính toán xong pValue, kiểm tra lại biên trước khi ghi vào ma trận P (global memory)
	if(row < P.height && column < P.width)
		P.elements[row*P.width+column] = pValue; // Ghi giá trị pValue vào vị trí tương ứng trong P
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_