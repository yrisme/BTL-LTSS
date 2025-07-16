Đề tài: TĂNG TỐC PHÉP NHÂN MA TRẬN SỬ DỤNG KỸ THUẬT CHIA KHỐI (TILED) VỚI CUDA 
TRIỂN KHAI NHÂN MA TRẬN SONG SONG BẰNG CUDA SỬ DỤNG KỸ THUẬT CHIA KHỐI 

Tổng quan về giải pháp 

Giải pháp nhân ma trận song song được triển khai bằng cách sử dụng kiến trúc CUDA của NVIDIA, tận dụng khả năng tính toán song song của GPU. Ý tưởng chính là chia ma trận kết quả P thành các khối (tiles) nhỏ hơn, và mỗi khối luồng (thread block) trên GPU sẽ chịu trách nhiệm tính toán một khối P này. Kỹ thuật chia khối (tiling) được áp dụng bằng cách sử dụng bộ nhớ chia sẻ (shared memory) để lưu trữ tạm thời các khối con của ma trận đầu vào M và N, nhằm giảm thiểu số lần truy cập vào bộ nhớ toàn cục (global memory) chậm chạp và tăng cường tái sử dụng dữ liệu. 

Cách ánh xạ bài toán lên kiến trúc CUDA (thread, block, grid): 

Grid: Toàn bộ ma trận kết quả P được ánh xạ vào một grid 2 chiều các khối luồng. Số lượng khối luồng theo chiều x của grid (gridSize.x) được tính bằng cách lấy trần của (chiều rộng P / TILE_WIDTH), và tương tự cho chiều y (gridSize.y = trần của (chiều cao P / TILE_WIDTH)). 

Block: Mỗi khối luồng (thread block) là một ma trận 2 chiều các luồng, có kích thước TILE_WIDTH x TILE_WIDTH. Mỗi khối luồng này chịu trách nhiệm tính toán một khối (tile) tương ứng có kích thước TILE_WIDTH x TILE_WIDTH trong ma trận kết quả P. 

Thread: Mỗi luồng (thread) trong một khối luồng chịu trách nhiệm tính toán một phần tử duy nhất trong khối P mà khối luồng đó quản lý. Tọa độ của luồng trong khối (threadIdx.x, threadIdx.y) kết hợp với tọa độ của khối trong grid (blockIdx.x, blockIdx.y) và TILE_WIDTH sẽ xác định phần tử P[row][column] mà luồng đó tính toán. 

Vai trò của shared memory trong việc lưu trữ các khối (tile): 

Bộ nhớ chia sẻ (__shared__) đóng vai trò cực kỳ quan trọng. Thay vì mỗi luồng đọc trực tiếp các phần tử từ ma trận M và N trong global memory cho mỗi phép nhân, các luồng trong một khối sẽ hợp tác để tải các khối (tiles) tương ứng của M và N vào hai mảng shared memory (tileMs và tileNs). Sau đó, các phép nhân và cộng dồn để tính toán các phần tử của khối P sẽ được thực hiện chủ yếu bằng cách truy cập dữ liệu từ shared memory, vốn nhanh hơn nhiều so với global memory. Điều này giúp: 

Giảm truy cập Global Memory: Mỗi phần tử trong các khối của M và N chỉ cần được tải từ global memory vào shared memory một lần cho mỗi "phase" tính toán khối P. 

Tăng tái sử dụng dữ liệu: Dữ liệu trong shared memory được tái sử dụng nhiều lần bởi các luồng trong cùng một khối. 

 

Hàm Kernel nhân ma trận chia khối (MatrixMulKernel) 

Đây là trái tim của việc tính toán song song trên GPU. 

 

Hình 4.1 Sơ đồ thuật toán song song phần Device 

Khai báo kernel (__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)): 

Từ khóa __global__ chỉ định rằng MatrixMulKernel là một hàm kernel sẽ được thực thi trên GPU và có thể được gọi từ mã host (CPU). Hàm nhận vào ba đối tượng kiểu Matrix (được định nghĩa trong matrixmul.h, có lẽ chứa con trỏ elements, width, height, pitch) đại diện cho ma trận M, N và ma trận kết quả P. 

Xác định chỉ số thread và block: 

“int tx = threadIdx.x; int ty = threadIdx.y; 

              int bx = blockIdx.x; int by = blockIdx.y;” 

Các biến này lưu trữ chỉ số của luồng hiện tại trong khối (tx, ty) và chỉ số của khối hiện tại trong grid (bx, by). 

Cấp phát và sử dụng shared memory: 

“__shared__ float tileMs[TILE_WIDTH][TILE_WIDTH]; 

  __shared__ float tileNs[TILE_WIDTH][TILE_WIDTH];” 

Hai mảng 2 chiều tileMs và tileNs được khai báo với từ khóa __shared__. Kích thước của chúng là TILE_WIDTH x TILE_WIDTH. Mỗi khối luồng sẽ có một bản sao riêng của hai mảng này. Chúng sẽ được dùng để lưu trữ các khối (tiles) của M và N. TILE_WIDTH được định nghĩa sẽ thay đổi tùy theo lúc test. 

Quá trình tải các khối (tile) từ global memory vào shared memory: 

Trong vòng lặp for(int i=0; i<ceilf(M.width/(float)TILE_WIDTH); i++): 
Mỗi luồng trong khối (ty, tx) chịu trách nhiệm tải một phần tử từ global memory của M và N vào vị trí tương ứng trong tileMs và tileNs. 

“int row = by * TILE_WIDTH + ty; 

int column = bx * TILE_WIDTH + tx; 

for(int i=0;i<ceilf(M.width/(float)TILE_WIDTH);i++){ 

if(row < M.height && (i*TILE_WIDTH + tx)<M.width)  // (A) Kiểm tra biên cho M 

    tileMs[ty][tx] = M.elements[row*M.width + i*TILE_WIDTH + tx];     // (B) Tải phần tử M vào shared memory 

else 

    tileMs[ty][tx] = 0; // (C) Gán 0 nếu ngoài biên 

if(column < N.width && (i*TILE_WIDTH + ty)<N.height) // (D) Kiểm tra biên cho N 

    tileNs[ty][tx] = N.elements[(i*TILE_WIDTH + ty)*N.width + column]; // (E) Tải phần tử N vào shared memory 

else 

    tileNs[ty][tx] = 0; // (F) Gán 0 nếu ngoài biên” 

(A) & (D): Các điều kiện if kiểm tra xem tọa độ truy cập có nằm trong giới hạn của ma trận M và N hay không. Điều này quan trọng khi kích thước ma trận không chia hết cho TILE_WIDTH. 

(B): Luồng (ty, tx) tải phần tử M[row][i*TILE_WIDTH + tx] vào tileMs[ty][tx]. row là hàng toàn cục của M mà luồng này liên quan (do khối luồng đang tính khối P tại hàng by), và i*TILE_WIDTH + tx là cột toàn cục của M. 

(E): Luồng (ty, tx) tải phần tử N[i*TILE_WIDTH + ty][column] vào tileNs[ty][tx]. i*TILE_WIDTH + ty là hàng toàn cục của N, và column là cột toàn cục của N mà luồng này liên quan. Lưu ý rằng cách các luồng tải tileNs có vẻ như đang tải một cột của N vào tileNs theo cách mà mỗi luồng tải một phần tử của cột đó. 

(C) & (F): Nếu tọa độ nằm ngoài biên, giá trị 0 được gán vào shared memory để không ảnh hưởng đến kết quả nhân. 

 

 Đồng bộ hóa các thread trong block (__syncthreads()): 

“__syncthreads(); // (G) 

… 

              __syncthreads(); // (H)” 

(G): Sau khi tất cả các luồng trong khối đã cố gắng tải xong phần tử của mình vào tileMs và tileNs cho "phase" i hiện tại, __syncthreads() được gọi. Lệnh này đảm bảo rằng tất cả các luồng trong khối phải đợi nhau tại đây cho đến khi toàn bộ dữ liệu của hai khối con đã được nạp hoàn chỉnh vào shared memory trước khi bất kỳ luồng nào bắt đầu thực hiện phép nhân. 

(H): Sau khi vòng lặp j (tính toán trên shared memory) kết thúc, __syncthreads() được gọi lại. Lệnh này đảm bảo rằng tất cả các luồng đã hoàn thành việc sử dụng dữ liệu trong tileMs và tileNs của "phase" i hiện tại, trước khi khối luồng chuyển sang "phase" i+1 (tức là tải các khối con tiếp theo của M và N). 

 

Thực hiện phép nhân các ma trận con trong shared memory: 

“float pValue = 0; 

... 

for(int j=0;j<TILE_WIDTH;j++) 

    pValue += tileMs[ty][j] * tileNs[j][tx]; // (I)” 

(I): Sau khi __syncthreads() (G) đảm bảo dữ liệu đã sẵn sàng trong shared memory, mỗi luồng thực hiện tính toán một phần của tích vô hướng. Luồng (ty, tx) tính pValue (sẽ là một phần tử của khối P mà nó phụ trách) bằng cách nhân hàng ty của tileMs với cột tx của tileNs và cộng dồn kết quả. Đây chính là phép nhân ma trận giữa hai khối con tileMs và tileNs đang nằm trong shared memory. Vòng lặp này chạy TILE_WIDTH lần. 

 

Ghi kết quả từ registers ra global memory: 

“if(row < P.height && column < P.width) // (J) Kiểm tra biên cho P 

    P.elements[row*P.width+column] = pValue; // (K) Ghi kết quả” 

Vòng lặp i (vòng lặp qua các "phase" hay các cặp khối con của M và N) chạy để tích lũy giá trị pValue hoàn chỉnh cho phần tử P[row][column]. 

(J): Sau khi vòng lặp i kết thúc, pValue chứa giá trị cuối cùng cho phần tử P[row][column]. Một lần nữa, kiểm tra biên được thực hiện để đảm bảo luồng chỉ ghi vào các vị trí hợp lệ của ma trận P. 

(K): Nếu trong biên, giá trị pValue (lưu trong thanh ghi của luồng) được ghi ra vị trí tương ứng P[row*P.width+column] trong global memory. 
 

Code CUDA kernel  

__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P) 

{ 

__shared__ float tileMs[TILE_WIDTH][TILE_WIDTH]; 

__shared__ float tileNs[TILE_WIDTH][TILE_WIDTH]; 

 

int tx = threadIdx.x; int ty = threadIdx.y; 

int bx = blockIdx.x; int by = blockIdx.y; 

 

int row = by * TILE_WIDTH + ty; 

int column = bx * TILE_WIDTH + tx; 

 

float pValue = 0; 

 

for(int i=0;i<ceilf(M.width/(float)TILE_WIDTH);i++){ 

if(row < M.height && (i*TILE_WIDTH + tx)<M.width) 

tileMs[ty][tx] = M.elements[row*M.width + i*TILE_WIDTH + tx]; 

else 

tileMs[ty][tx] = 0; 

if(column < N.width && (i*TILE_WIDTH + ty)<N.height) 

tileNs[ty][tx] = N.elements[(i*TILE_WIDTH + ty)*N.width + column]; 

else 

tileNs[ty][tx] = 0; 

 

__syncthreads(); 

 

for(int j=0;j<TILE_WIDTH;j++) 

pValue += tileMs[ty][j] * tileNs[j][tx]; 

 

__syncthreads(); 

} 

if(row < P.height && column < P.width) 

P.elements[row*P.width+column] = pValue; 

} 

 

Hàm Host gọi Kernel (MatrixMulOnDevice trong matrixmul.cu) 

Hàm này chịu trách nhiệm quản lý dữ liệu, cấu hình và khởi chạy kernel trên GPU. 

 

Hình 4.2 Sơ đồ thuật toán song song phần Host 

Cấp phát bộ nhớ trên Host (CPU) và Device (GPU): 

Host: Hàm AllocateMatrix (trong main) cấp phát bộ nhớ cho M, N, P trên CPU bằng malloc. 

Device: Hàm AllocateDeviceMatrix (được gọi trong MatrixMulOnDevice) cấp phát bộ nhớ trên GPU cho Md, Nd, Pd bằng cudaMalloc. 

“Matrix Md = AllocateDeviceMatrix(M); 

... 

Matrix Nd = AllocateDeviceMatrix(N); 

... 

Matrix Pd = AllocateDeviceMatrix(P);” 

Sao chép dữ liệu từ Host sang Device (cudaMemcpyHostToDevice): 

Hàm CopyToDeviceMatrix được sử dụng để sao chép dữ liệu từ ma trận M, N trên host sang Md, Nd trên device. 

“CopyToDeviceMatrix(Md, M); 

  CopyToDeviceMatrix(Nd, N);” 

Bên trong CopyToDeviceMatrix: 

“cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);” 

Cấu hình Grid và Block: 

“dim3 gridSize, blockSize; 

blockSize.x = blockSize.y = TILE_WIDTH; blockSize.z = 1; 

gridSize.x = ceil(P.width/(float)blockSize.x); 

gridSize.y = ceil(P.height/(float)blockSize.y); 

gridSize.z = 1;” 

blockSize: Kích thước của mỗi khối luồng được đặt là TILE_WIDTH x TILE_WIDTH  

gridSize: Kích thước của grid (số lượng khối luồng) được tính toán dựa trên kích thước của ma trận P và TILE_WIDTH, đảm bảo có đủ khối luồng để bao phủ toàn bộ ma trận P. Hàm ceil được dùng để làm tròn lên. 

Gọi Kernel: 

“MatrixMulKernel<<<gridSize, blockSize>>>(Md, Nd, Pd);” 

Đây là lệnh khởi chạy kernel MatrixMulKernel trên GPU. gridSize và blockSize xác định cấu hình thực thi. Md, Nd, Pd là các con trỏ tới dữ liệu ma trận trên device. 

Đồng bộ hóa (cudaDeviceSynchronize() hoặc thông qua cudaEventSynchronize): 

Trong đoạn code này, việc đồng bộ hóa được thực hiện ngầm thông qua cudaEventSynchronize(stop); sau khi đo thời gian. Lệnh này đảm bảo rằng CPU sẽ đợi cho đến khi kernel MatrixMulKernel (và các lệnh CUDA trước đó liên quan đến event stop) hoàn thành việc thực thi trên GPU. 
“cudaEventRecord(stop, 0); 

cudaEventSynchronize(stop); // Chờ kernel hoàn thành” 

Sao chép kết quả từ Device về Host (cudaMemcpyDeviceToHost): 

Sau khi kernel hoàn thành, hàm CopyFromDeviceMatrix được gọi để sao chép ma trận kết quả Pd từ device về ma trận P trên host. 

“CopyFromDeviceMatrix(P, Pd);” 

Bên trong CopyFromDeviceMatrix:  

“cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);” 

Giải phóng bộ nhớ: 

Device: Hàm FreeDeviceMatrix được dùng để giải phóng bộ nhớ đã cấp phát trên GPU cho Md, Nd, Pd bằng cudaFree. 

“FreeDeviceMatrix(&Md); 

 FreeDeviceMatrix(&Nd); 

 FreeDeviceMatrix(&Pd);” 

Host: Hàm FreeMatrix (trong main) giải phóng bộ nhớ đã cấp phát trên CPU cho M, N, P bằng free. 

Code Host gọi Kernel 

void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P) 

{ 

    Matrix Md = AllocateDeviceMatrix(M); 

    CopyToDeviceMatrix(Md, M); 

    Matrix Nd = AllocateDeviceMatrix(N); 

    CopyToDeviceMatrix(Nd, N); 

 

    Matrix Pd = AllocateDeviceMatrix(P); 

    CopyToDeviceMatrix(Pd, P);  

 

    dim3 gridSize, blockSize; 

    blockSize.x = blockSize.y = TILE_WIDTH; blockSize.z = 1; 

    gridSize.x = ceil(P.width/(float)blockSize.x); 

    gridSize.y = ceil(P.height/(float)blockSize.y); 

    gridSize.z = 1; 

 

    cudaEvent_t start, stop; 

    cudaEventCreate(&start); 

    cudaEventCreate(&stop); 

    cudaEventRecord(start, 0); 

 

    MatrixMulKernel<<<gridSize, blockSize>>>(Md, Nd, Pd); 

 

    cudaEventRecord(stop, 0); 

    cudaEventSynchronize(stop); 

 

    float gpu_time = 0.0f; 

    cudaEventElapsedTime(&gpu_time, start, stop); 

    printf("🚀 GPU Time: %.4f ms\n", gpu_time); 

 

    cudaEventDestroy(start); 

    cudaEventDestroy(stop); 

 

    CopyFromDeviceMatrix(P, Pd);  

 

    FreeDeviceMatrix(&Md); 

    FreeDeviceMatrix(&Nd); 

    FreeDeviceMatrix(&Pd); 

} 
