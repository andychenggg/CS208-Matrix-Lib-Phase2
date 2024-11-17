#include <stdio.h>
#include "matrix.h"
#include <cblas.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>

#define TIME_START start=clock();
#define TIME_END(NAME) end=clock(); \
             printf("%s%s%.3f%s\n", NAME, ": duration = ", ((float)end - start)/CLK_TCK, "s");

#define TIME_openmp_START omp_start = omp_get_wtime();
#define TIME_openmp_END(NAME) omp_end = omp_get_wtime(); \
                            printf("%s%s%.3f%s\n", NAME, ": duration = ", (omp_end - omp_start), "s");
void load(const float* pf1, float *pf2, size_t length){
    for(size_t i = 0; i<length; i++){
        *pf2 = *pf1;
        pf2++;
        pf1++;
    }
}
void load1(const float* pf1, float *pf2, size_t length){
    for(size_t i = 0; i<length/16; i++){
        _mm512_storeu_ps(pf2, _mm512_loadu_ps(pf1));
        pf1+=16;
        pf2+=16;
    }
}
int main() {



//
//
//
//    TIME_START
//
//    TIME_START
//    matmul_improved_2_21(p1, p2, 64);
//    TIME_END("improved2_21")
//
//    TIME_START
//    matmul_improved2_3_1_avx2(p1, p2);
//    TIME_END("improved2_3_1_avx2")

//    TIME_START
//    matmul_improved2_3_1_avx512(p1, p2);
//    TIME_END("improved2_3_1_avx512")



//
//    TIME_START
//    matmul_improved2_3_1_avx512_2(p1, p2);
//    TIME_END("improved2_3_1_avx512_2")
//    TIME_START
//    matmul_improved2_3_2(p1, p2);
//    TIME_END("improved2_3_2")



//
//    TIME_START
//    matmul_improved2_3_11(p1, p2);
//    TIME_END("improved2_3_11")


//    TIME_START
//    matmul_improved2_3_2(p1, p2);
//    TIME_END("improved2_3_2")

//    TIME_START
//    matmul_improved2_3_31(p1, p2);
//    TIME_END("improved2_3_31")

//    TIME_START
//    matmul_improved2_3_32(p1, p2);
//    TIME_END("improved2_3_32")

//    TIME_START
//    matmul_improved2_3_33(p1, p2);
//    TIME_END("improved2_3_33")

//    TIME_START
//    matmul_improved2_3_4(p1, p2, 16);
//    TIME_END("improved2_3_4, 16:")
//
//    TIME_START
//    matmul_improved2_3_4(p1, p2, 32);
//    TIME_END("improved2_3_4, 32:")
//
//    TIME_START
//    matmul_improved2_3_4(p1, p2, 64);
//    TIME_END("improved2_3_4, 64:")
//

//    TIME_START
//    matmul_improved_2_2(p1, p2, 256);
//    TIME_END("matmul_improved_2_2")

//    TIME_START
//    matmul_improved2_3_4(p1, p2, 16);
//    TIME_END("improved2_3_4, 16:")
//
//
//    TIME_START
//    matmul_improved2_3_4(p1, p2, 128);
//    TIME_END("improved2_3_4, 128:")
    double omp_start = 0., omp_end = 0.;
    const long long  si = 4000;
    size_t size = si;
    Matrix *p1 = create_an_identity_matrix(size);
    Matrix *p2 = create_an_identity_matrix(size);
    //程序预热
    printf("Programs \"warm up\" begin!\n");
    set_Matrix_same_value(p1, 5);
    set_Matrix_same_value(p2, 5);
    printf("Programs \"warm up\" end!\n");
    const long long M = si;
    const long long N1 = si;
    const long long K = si;
    const float alpha = 1;
    const float beta = 0;
    const long long lda = K;
    const long long ldb = N1;
    const long long ldc = N1;
//    const float *A1 = (float *) malloc(si * si * sizeof(float)); //M * K
//    const float *B1 = (float *) malloc(si * si * sizeof(float)); //K*N
//    float *C1 = (float *) malloc(si * si * sizeof(float)); //M * N

    TIME_openmp_START
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N1, K, alpha, p1->pointer_of_data, lda, p1->pointer_of_data, ldb, beta, p2->pointer_of_data, ldc);
    TIME_openmp_END("openblas")

    TIME_openmp_START
    matmul_plain(p1, p2);
    TIME_openmp_END("plain")



//    TIME_openmp_START
//    matmul_improved2_1(p1, p2);
//    TIME_openmp_END("improved2_1")
//
//    TIME_openmp_START
//    matmul_improved_2_2(p1, p2, 64);
//    TIME_openmp_END("improved2_2")

    TIME_openmp_START
    matmul_improved2_3_4(p1, p2, 256);
    TIME_openmp_END("improved2_3_4, 256:")

//    TIME_openmp_START
//    matmul_improved2_3_1_avx512(p1, p2);
//    TIME_openmp_END("improved2_3_1_avx512")

//    TIME_openmp_START
//    matmul_improved2_4_2(p1, p2);
//    TIME_openmp_END("matmul_strassen")

//    TIME_START
//    matmul_improved2_3_4(p1, p2, 512);
//    TIME_END("improved2_3_4, 512:")
//
//    TIME_START
//    matmul_improved2_3_4(p1, p2, 512);
//    TIME_END("improved2_3_4, 512:")
////
//    TIME_START
//    matmul_improved2_3_4(p1, p2, 1024);
//    TIME_END("improved2_3_4, 1024:")
//
//    TIME_START
//    matmul_improved2_3_4(p1, p2, 2048);
//    TIME_END("improved2_3_4, 2048:")
//
//    TIME_START
//    matmul_improved2_3_4(p1, p2, 2048);
//    TIME_END("improved2_3_4, 2048:")
//
//    float *f = (float*) malloc(1024000000 * sizeof(float ));
//
//    float *f1 = (float*) malloc(1024000000 * sizeof(float ));
//
//    float *f2 = (float*) malloc(1024000000 * sizeof(float ));
//    TIME_START
//    load(f, f1, 1024000000);
//    TIME_END("l")
//    TIME_START
//    load1(f, f2, 1024000000);
//    TIME_END("l1")

}

//#define NB_L_STRIDE 8
//
//void mul1() {
//    __m512d res_ptr_8[NB_L_STRIDE], B_ptr_8;
//    for (unsigned int res_l = 0; res_l < res_matrix->nb_l; res_l += NB_L_STRIDE) {
//        for (unsigned int res_c = 0; res_c < res_matrix->nb_c; res_c += 8) {
//            for (unsigned int i = 0; i < NB_L_STRIDE; i++)
//                res_ptr_8[i] = _mm512_setzero_pd();
//            for (unsigned int offset_A_c = 0; offset_A_c < A->nb_c; offset_A_c++) {
//// compute values from res_matrix[res_l, res_c] to [res_l, res_c+7]
//// on the res_l th line of A pick values one at a time
//// at coordinates A[res_l, offset_A_c].
//// Broadcast this value eight times into a mm512 vector
//// and perform a dot product with the 8 values found in
//// B from coordinates [offset_A_c, res_c] to [offset_A_c, res_c + 7]
//                B_ptr_8 = _mm512_load_pd(&B->values[offset_A_c * B->nb_c + res_c]);
//                for (unsigned int i = 0; i < NB_L_STRIDE; i++)
//                    res_ptr_8[i] = _mm512_fmadd_pd(
//                            _mm512_set1_pd(A->values[(res_l + i) * A->nb_c + offset_A_c]),
//                            B_ptr_8,
//                            res_ptr_8[i]);
//
//            }
//            for (unsigned int i = 0; i < NB_L_STRIDE; i++)
//                _mm512_store_pd(
//                        &res_matrix->values[(res_l + i) * res_matrix->nb_c + res_c],
//                        res_ptr_8[i]);
//        }
//    }
//};
