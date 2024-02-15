#include <bits/stdc++.h>
#include <fstream>
#include <chrono>
#include <time.h>
#include "constants.h"
using namespace std;

vector<vector<double>> A(N, vector<double>(N, 0.0));
vector<vector<double>> PA(N, vector<double>(N, 0.0));
vector<vector<double>> LU(N, vector<double>(N, 0.0));
vector<vector<double>> residual(N, vector<double>(N, 0.0));
double temp_A[N][N];
double L[N][N], U[N][N];
int P[N][N];
int pi[N];

void inputMatrix(){
    random_device rd;
    default_random_engine generator(rd());
    uniform_real_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i][j] = distribution(generator);
        }
    }
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

void LUdecompose(){
    ofstream fout;
    #ifdef TIMING
    fout.open(DEBUG_OUT_FILE_2, ios::app);
    double total_A_time = 0.0;
    #endif
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
        for(int i=k+1; i<N; i++){
            L[i][k] = (temp_A[i][k]*1.0)/U[k][k];
            U[k][i] = temp_A[k][i];
        }
        #ifdef TIMING
        auto start_A_time = chrono::high_resolution_clock::now();
        #endif
        for(int i=k+1; i<N; i++){
            for(int j=k+1; j<N; j++){
                temp_A[i][j] -= L[i][k]*U[k][j];
            }
        }
        #ifdef TIMING
        auto end_A_time = chrono::high_resolution_clock::now();
        double time_taken_A = chrono::duration_cast<chrono::nanoseconds>(end_A_time - start_A_time).count();
        total_A_time += time_taken_A;
        fout << "A time: " << time_taken_A << " ns" << endl;
        #endif
    }
    #ifdef TIMING
    fout << "Total A time: " << total_A_time << " ns" << endl;
    fout.close();
    #endif
    for(int i=0; i<N; i++){
        P[i][pi[i]] = 1;
    }

    auto end_time = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    fout.open(LOG_OUT_FILE, ios::app);
    fout << "-----------------------------------------------\n";
    fout << "N=" << N << ", Sequential,"<< " Threads=1, " << time_taken << " ms" << endl;
    #ifdef TIMING
    fout << "Total A update time SEQ: " << total_A_time << " ns" << endl;
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