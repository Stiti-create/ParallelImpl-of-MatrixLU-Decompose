#include <bits/stdc++.h>
#include <fstream>
#include <chrono>
#include <time.h>
#include <pthread.h>
#include "constants.h"
using namespace std;

vector<int> pi(N, 0);
vector<vector<double>> A(N, vector<double>(N, 0.0));
vector<vector<double>> L(N, vector<double>(N, 0.0));
vector<vector<double>> U(N, vector<double>(N, 0.0));
vector<vector<int>> P(N, vector<int>(N, 0));
vector<vector<double>> PA(N, vector<double>(N, 0.0));
vector<vector<double>> LU(N, vector<double>(N, 0.0));
vector<vector<double>> residual(N, vector<double>(N, 0.0));
vector<vector<double>> temp_A(N, vector<double>(N, 0.0));

struct pthread_args{
    int k;
    int thread_id;
    int temp_k;
    int left_bound;
    int right_bound;
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

void *LUdecompose_work_parallel(void* pthread_args){
    struct pthread_args* args = (struct pthread_args*)pthread_args;
    int k = args->k;
    int temp_k = args->temp_k;
    int id = args->thread_id;
    int l_1 = args->left_bound;
    int r_1 = args->right_bound;
    for(int i=l_1; i<r_1; i++){
        swap(temp_A[k][i], temp_A[temp_k][i]);
    }
    
    int l_2 = (int)(ceil(k/(1.0*PTHREAD_COUNT)))*(id);
    int r_2 = (int)(ceil(k/(1.0*PTHREAD_COUNT)))*(id+1);
    if(id == PTHREAD_COUNT-1){
        r_2 = k;
    }
    for(int j=l_2; j<r_2; j++){
        swap(L[k][j], L[temp_k][j]);
    }

    int l_3 = k+1 + (int)(ceil((N-k-1)/(1.0*PTHREAD_COUNT)))*(id); // TODO: correct this
    int r_3 = k+1 + (int)(ceil((N-k-1)/(1.0*PTHREAD_COUNT)))*(id+1); //TODO: correct this
    if(id == PTHREAD_COUNT-1){
        r_3 = N;
    }
    #ifdef DEBUG
        ofstream fout;
        fout.open(DEBUG_OUT_FILE, ios::app);
        fout << "Thread " << id  << " k: " << k << " l_3: " << l_3 << " r_3: " << r_3 << endl;
    #endif
    for(int ind=l_3; ind<r_3; ind++){
        L[ind][k] = (temp_A[ind][k]*1.0)/U[k][k];
        U[k][ind] = temp_A[k][ind];
    }
    for(int i=l_3; i<r_3; i++){
        for(int j=k+1; j<N; j++){
            temp_A[i][j] -= L[i][k]*U[k][j];
        }
    }
    pthread_exit(NULL);
}

void LUdecompose(){
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
        if(maxi == 0.0){
            perror("Singular matrix");
        }
        U[k][k] = temp_A[k][k];
        swap(pi[k], pi[temp_k]);
        for(int num = 0; num < PTHREAD_COUNT; num++){
            args[num].k = k;
            args[num].thread_id = num;
            args[num].left_bound = (N/PTHREAD_COUNT)*num;
            args[num].right_bound = (N/PTHREAD_COUNT)*(num+1);
            if(num == PTHREAD_COUNT-1){
                args[num].right_bound = N;
            }
            args[num].temp_k = temp_k;
            pthread_create(&threads[num], NULL, &LUdecompose_work_parallel, (void*)&args[num]);
        }

        
        // temp_A[k].swap(temp_A[temp_k]);
        // for(int i=0; i<k; i++){
        //     swap(L[k][i], L[temp_k][i]);
        // }

        // for(int i=k+1; i<N; i++){
        //     L[i][k] = (temp_A[i][k]*1.0)/U[k][k];
        //     U[k][i] = temp_A[k][i];
        // }
        // for(int i=k+1; i<N; i++){
        //     for(int j=k+1; j<N; j++){
        //         temp_A[i][j] -= L[i][k]*U[k][j];
        //     }
        // }

        for(int num = 0; num < PTHREAD_COUNT; num++){
            pthread_join(threads[num], NULL);
        }
    }

    // can be done parallelly in future, let it be sequential for now
    for(int i=0; i<N; i++){
        P[i][pi[i]] = 1;
    }
    auto end_time = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    ofstream fout;
    fout.open(LOG_OUT_FILE, ios::app);
    fout << "N: " << N << ", Parallel (pthread): " << time_taken << " ms" << endl;
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
    fout.open(LU_VERIFY_OUT, ios::app);
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