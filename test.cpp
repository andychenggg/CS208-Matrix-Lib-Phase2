//#include <cblas.h>
//#include <iostream>
//#include <omp.h>
//using namespace std;
//int main(){
//    const int M = 4;//A的行数，C的行数
//    const int N = 2;//B的列数，C的列数
//    const int K = 3;//A的列数，B的行数
//    const double alpha = 1;
//    const double beta = 0;
//    const double A[K*M] = { 1,2,3,4,5,6,7,8,9,8,7,6 };
//    const double B[K*N] = { 5,4,3,2,1,0 };
//    double C[M*N] = {0};
//    int lda = K;
//    int ldb = N;
//    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, lda, B,ldb, beta, C, N);
//    for(int i=0;i<M;i++){
//        for(int j=0;j<N;j++){
//            cout<<C[i*N+j]<<"\t";
//        }
//        cout<<endl;
//    }
//    return 0;
//}
//int main(){
//    int a[4];
//    for(auto i: a){
//        cout<<i<<" ";
//    }
//}
