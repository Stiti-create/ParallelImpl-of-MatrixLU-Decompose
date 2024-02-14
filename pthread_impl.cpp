#include <bits/stdc++.h>
#include <fstream>
#include <chrono>
#include <time.h>
#include <pthread.h>
#include "constants.h"
using namespace std;

int pi[N];
int P[N][N];
double A[N][N];
double L[N][N], U[N][N];
double* temp_A[N];
double PA[N][N];
double LU[N][N];
double residual[N][N];
double l[N], u[N];

struct pthread_args{
    int k;
    int thread_id;
    int temp_k;
};

void inputMatrix(){
    ifstream fin;
    fin.open(INPUT_MATRIX_FILE);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            fin >> A[i][j];
            if(fin.eof()) break;
        }
    }
    fin.close();
}

void initOutputs(){
    for(int i = 0; i < N; i++){
        temp_A[i] = (double *)malloc(N * sizeof(double));
        for(int j = 0; j < N; j++){
            U[i][j] = A[i][j];
            L[i][j] = A[i][j];
            temp_A[i][j] = A[i][j];
        }
    }
    for(int i = 0; i < N; i++){
        pi[i] = i;
        L[i][i] = 1.0;
    }
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            if(i>j){
                U[i][j] = 0;
            }
            if(i<j){
                L[i][j] = 0;
            }
        }
    }
}

pair<int,int> getBounds(int id, int num_threads, int num_iter){
    int q = num_iter/num_threads;
    int c = int(ceil(num_iter/(1.0*num_threads)));
    int r = num_iter%num_threads;
    int base = r*c;
    if(id<r){
        return {id*c, (id+1)*c};
    }
    else{
        return {base+(id-r)*q, base+(id+1-r)*q};
    }
}

void *parallel_compute(void* pthread_args){
    struct pthread_args* args = (struct pthread_args*)pthread_args;
    int k = args->k;
    int temp_k = args->temp_k;
    int id = args->thread_id;

    pair<int,int> bounds = getBounds(id, PTHREAD_COUNT, N-k-1);
    int left = k+1 + bounds.first;
    int right = k+1 + bounds.second;

    for(int i=left; i<right; i++){
        L[i][k] = temp_A[i][k]/U[k][k];
        U[k][i] = temp_A[k][i];
        double x = (temp_A[i][k]/U[k][k]);
        for(int j=k+1; j<N; j+=8){
            temp_A[i][j] -= (x*temp_A[k][j]);
            if(j+1<N) temp_A[i][j+1] -= (x*temp_A[k][j+1]);
            if(j+2<N) temp_A[i][j+2] -= (x*temp_A[k][j+2]);
            if(j+3<N) temp_A[i][j+3] -= (x*temp_A[k][j+3]);
            if(j+4<N) temp_A[i][j+4] -= (x*temp_A[k][j+4]);
            if(j+5<N) temp_A[i][j+5] -= (x*temp_A[k][j+5]);
            if(j+6<N) temp_A[i][j+6] -= (x*temp_A[k][j+6]);
            if(j+7<N) temp_A[i][j+7] -= (x*temp_A[k][j+7]);
        }
    }
    
    pthread_exit(NULL);
}
 

void LUdecompose(){
    ofstream fout;

    #ifdef TIMING
    fout.open(DEBUG_OUT_FILE_3, ios::app);
    double total_A_time = 0.0;
    #endif

    pthread_args args[PTHREAD_COUNT];
    pthread_t threads[PTHREAD_COUNT];

    auto start_time = chrono::high_resolution_clock::now();
    for (int k=0; k<N; k++){
        double maxi = 0.0;
        int temp_k = k;
        for (int i=k; i<N; i++){
            if (maxi < temp_A[i][k]){
                maxi = temp_A[i][k];
                temp_k = i;
            }
        }
        #ifdef DEBUG
        if(maxi == 0.0){
            perror("Singular matrix");
        }
        #endif
        U[k][k] = temp_A[temp_k][k];
        swap(pi[k], pi[temp_k]);
        swap(temp_A[k], temp_A[temp_k]);

        for(int i=0; i<k; i++){
            swap(L[k][i], L[temp_k][i]);
        }

        #ifdef TIMING
        auto inner_start_time = chrono::high_resolution_clock::now();
        #endif

        for(int num = 0; num < PTHREAD_COUNT; num++){
            args[num].k = k;
            args[num].thread_id = num;
            args[num].temp_k = temp_k;
            pthread_create(&threads[num], NULL, &parallel_compute, (void*)&args[num]);
        }
        for (int num = 0; num < PTHREAD_COUNT; num++){
            pthread_join(threads[num], NULL);
        }
        #ifdef TIMING
        auto inner_end_time = chrono::high_resolution_clock::now();
        
        double inner_time_taken = chrono::duration_cast<chrono::nanoseconds>(inner_end_time - inner_start_time).count();
        total_A_time += inner_time_taken;
        
        fout << "Inner Parallel Time: " << inner_time_taken << " ns\n" << endl;
        #endif
    }

    #ifdef TIMING
        fout << "Total A Time: " << total_A_time << " ns" << endl;
        fout.close();
    #endif

    for(int i=0; i<N; i++){
        P[i][pi[i]] = 1;
    }
    auto end_time = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    fout.open(LOG_OUT_FILE, ios::app);
    fout << "-----------------------------------------------\n";
    fout << "N=" << N << ", Pthreads,"<< " Threads=" <<PTHREAD_COUNT <<", " << time_taken << " ms" << endl;
    #ifdef TIMING
    fout << "Total A update time PTH: " << total_A_time << " ns" << endl;
    #endif
    fout.close();
    return;
}

void verifyLU(){
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            PA[i][j] = 0.0;
            LU[i][j] = 0.0;
            for(int k=0; k<N; k++){
                PA[i][j] += P[i][k]*A[k][j];
                LU[i][j] += L[i][k]*U[k][j];
            }
        }
    }

    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            residual[i][j] = PA[i][j] - LU[i][j];
        }
    }

    #ifdef DEBUG_LU_VERIFY
    ofstream fout;
    fout.open(LU_VERIFY_OUT);
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            fout << residual[i][j] << " ";
        }
        fout << endl;
    }
    fout.close();
    #endif

    double norm = 0.0;
    for(int j=0; j<N; j++){
        double col_eucledian_norm = 0.0;
        for(int i=0; i<N; i++){
            col_eucledian_norm += pow(residual[i][j], 2);
        }
        norm += sqrt(col_eucledian_norm);
    }
    cout << "L-2,1 Norm of residual: " << norm << endl;

}

int main(){
    inputMatrix();
    initOutputs();
    LUdecompose();
    verifyLU();
}