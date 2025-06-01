#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>          // Đo thời gian CPU
#include <cuda_runtime.h> // Đo thời gian GPU

#include "matrixmul_kernel.cu"


extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
int ReadFile(Matrix* M, char* file_name);
void WriteFile(Matrix M, char* file_name);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);
bool CompareMatrices(Matrix A, Matrix B);
void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P);
int ReadParamsFile(int* params, char* file_name, int num_params);
void computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB) {
    for (unsigned int i = 0; i < hA; ++i) {
        for (unsigned int j = 0; j < wB; ++j) {
            double sum = 0;
            for (unsigned int k = 0; k < wA; ++k) {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = (float)sum;
        }
    }
}


int main(int argc, char** argv) {

	Matrix  M;
	Matrix  N;
	Matrix  P;
	int errorM = 0, errorN = 0;
	
	srand(52);
	
	if(argc != 5 && argc != 4) 
	{
	 // Cấp phát và khởi tạo các ma trận
		M  = AllocateMatrix(rand() % 1024, rand() % 1024, 1);
		N  = AllocateMatrix(M.width, rand() % 1024, 1);
		P  = AllocateMatrix(M.height, N.width, 0);
	}
	else
	{
		// Cấp phát và đọc các ma trận từ disk
		int* params = (int*)malloc(3 * sizeof(int));
		unsigned data_read = ReadParamsFile(params, argv[1], 3);
		if(data_read != 3){
			printf("Error reading parameter file\n"); 
			return 1;
		}

		M  = AllocateMatrix(params[0], params[1], 0);
		N  = AllocateMatrix(params[1], params[2], 0);		
		P  = AllocateMatrix(params[0], params[2], 0);
		unsigned sizeM = ReadFile(&M, argv[2]);
		unsigned sizeN = ReadFile(&N, argv[3]);
		if( (sizeM != M.height * M.width) || (sizeN != N.height * N.width) )
		{
			printf("Error reading input files %d, %d\n", errorM, errorN); 
			return 1;
		}
	}

	// Thực hiện M * N trên thiết bị (GPU)
    MatrixMulOnDevice(M, N, P);
    
    printf("GPU computation complete\n");
    // thực hiện phép nhân ma trận trên CPU để so sánh
    Matrix reference = AllocateMatrix(P.height, P.width, 0);
        clock_t cpu_start = clock();
    computeGold(reference.elements, M.elements, N.elements, M.height, M.width, N.width);
    clock_t cpu_end = clock();
    double cpu_time = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("⏱️  CPU Time: %.4f ms\n", cpu_time);

        
    printf("CPU computation complete\n"); 
   // kiểm tra xem kết quả từ thiết bị có tương đương với giải pháp mong đợi không
    bool res = CompareMatrices(reference, P);
    printf("Test %s\n", res ? "PASSED" : "FAILED"); 
    
    if(argc == 5)
    {
		WriteFile(P, argv[4]);
	}
	else if(argc == 2)
	{
	    WriteFile(P, argv[1]);
	}   

	// Giải phóng các ma trận
    FreeMatrix(&M);
    FreeMatrix(&N);
    FreeMatrix(&P);
	return 0;
}


void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P)
{
    // Đưa M và N lên thiết bị
    Matrix Md = AllocateDeviceMatrix(M);
    CopyToDeviceMatrix(Md, M);
    Matrix Nd = AllocateDeviceMatrix(N);
    CopyToDeviceMatrix(Nd, N);

    // Cấp phát P trên thiết bị
    Matrix Pd = AllocateDeviceMatrix(P);
    CopyToDeviceMatrix(Pd, P); // Xóa bộ nhớ (khởi tạo P trên device)

    // Thiết lập cấu hình 
    dim3 gridSize, blockSize;
    blockSize.x = blockSize.y = TILE_WIDTH; blockSize.z = 1;
    gridSize.x = ceil(P.width/(float)blockSize.x);
    gridSize.y = ceil(P.height/(float)blockSize.y);
    gridSize.z = 1;


    
        // Tạo event đo thời gian
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Khởi chạy các luồng tính toán trên thiết bị!
    MatrixMulKernel<<<gridSize, blockSize>>>(Md, Nd, Pd);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("🚀 GPU Time: %.4f ms\n", gpu_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    // Đọc P từ thiết bị
    CopyFromDeviceMatrix(P, Pd); 

    // Giải phóng các ma trận trên thiết bị
    FreeDeviceMatrix(&Md);
    FreeDeviceMatrix(&Nd);
    FreeDeviceMatrix(&Pd);
}

// Cấp phát một ma trận trên thiết bị có cùng kích thước với M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Cấp phát một ma trận trên thiết bị với kích thước height*width
// Nếu init == 0, khởi tạo tất cả bằng không.
// Nếu init == 1, thực hiện khởi tạo ngẫu nhiên.
// Nếu init == 2, khởi tạo các tham số ma trận, nhưng không cấp phát bộ nhớ
Matrix AllocateMatrix(int height, int width, int init)
{
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;
    
    // không cấp phát bộ nhớ với tùy chọn 2
    if(init == 2)
		return M;
		
	M.elements = (float*) malloc(size*sizeof(float));

	for(unsigned int i = 0; i < M.height * M.width; i++)
	{
		M.elements[i] = (init == 0) ? (0.0f) : (rand()*3 / (float)RAND_MAX);
	}
    return M;
}	

// Sao chép một ma trận từ host sang ma trận trên thiết bị.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    Mdevice.height = Mhost.height;
    Mdevice.width = Mhost.width;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, 
					cudaMemcpyHostToDevice);
}

// Sao chép một ma trận từ thiết bị sang ma trận trên host.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.width * Mdevice.height * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, 
					cudaMemcpyDeviceToHost);
}

// Giải phóng một ma trận trên thiết bị.
void FreeDeviceMatrix(Matrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Giải phóng một ma trận trên Host
void FreeMatrix(Matrix* M)
{
    free(M->elements);
    M->elements = NULL;
}

// Đọc một floating point matrix từ file
// Trả về số phần tử đã đọc nếu bằng M.height * M.width, ngược lại trả về giá trị khác (logic này cần xem lại, hàm trả về data_read)
int ReadFile(Matrix* M, char* file_name)
{
    unsigned int data_read = M->width * M->height;
    FILE* input = fopen(file_name, "r");
    for (unsigned i = 0; i < data_read; i++) 
        fscanf(input, "%f", &(M->elements[i]));
    return data_read;
}

// Đọc các tham số của ma trận đầu vào
int ReadParamsFile(int* params, char* file_name, int num_params)
{
    FILE* input = fopen(file_name, "r");
    for (unsigned i = 0; i < num_params; i++) 
        fscanf(input, "%d", &(params[i]));
    return num_params;
}

// Ghi một floating point matrix M.width * M.height ra file 
void WriteFile(Matrix M, char* file_name)
{
    unsigned int size = M.width * M.height;
    FILE* output = fopen(file_name, "w");
    for (unsigned i = 0; i < size; i++) {
        fprintf(output, "%f ", M.elements[i]);
    }
}

// trả về true nếu và chỉ nếu A và B có các phần tử giống nhau theo cùng thứ tự
bool CompareMatrices(Matrix A, Matrix B) {
    unsigned int size = A.width * A.height;

    if ( (A.width != B.width) || (A.height != B.height) )
        return false;

    for (unsigned i = 0; i < size; i++)
        if (abs(A.elements[i] - B.elements[i]) > 0.0001f)
            return false;
    return true;
}