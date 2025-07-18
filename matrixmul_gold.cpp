#include <stdlib.h>
extern "C"
void computeGold( float*, const float*, const float*, unsigned int, unsigned int, unsigned int);


//! @tham số C      Dữ liệu tham chiếu (kết quả), đã được cấp phát trước và sẽ được hàm này tính toán.
//! @tham số A      Ma trận đầu vào A.
//! @tham số B      Ma trận đầu vào B.
//! @tham số hA     Chiều cao của ma trận A.
//! @tham số wB     Chiều rộng của ma trận B.
// Lưu ý: Tham số thứ năm trong khai báo hàm là wA (chiều rộng của ma trận A / chiều cao của ma trận B),
void
computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j) {
            double sum = 0;
            for (unsigned int k = 0; k < wA; ++k) { // wA là chiều rộng của A và chiều cao của B
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = (float)sum;
        }
}
