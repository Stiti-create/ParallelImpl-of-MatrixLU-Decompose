#include <bits/stdc++.h>
#include <fstream>
#include "constants.h"
using namespace std;

vector<int> pi(N, 0);
vector<vector<double>> A(N, vector<double>(N, 0.0));
vector<vector<double>> L(N, vector<double>(N, 0.0));
vector<vector<double>> U(N, vector<double>(N, 0.0));
vector<vector<int>> P(N, vector<int>(N, 0.0));

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
        pi[i] = i;
        L[i][i] = 1;
    }
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            if(i>j){
                U[i][j] = 0;
            }
            if(i<j){
                L[i][j] = 0;
            }
            P[i][j] = 0;
        }
    }
}

void LUdecompose(){
    for (int k=0; k<N; k++){
        double maxi = 0.0;
        int temp_k = k;
        for (int i=k; i<N; i++){
            if (maxi < abs(A[i][k])){
                maxi = abs(A[i][k]);
                temp_k = i;
            }
        }
        if(maxi == 0.0){
            perror("Singular matrix");
        }
        swap(pi[k], pi[temp_k]);
        A[k].swap(A[temp_k]);
        for(int i=0; i<k; i++){
            swap(L[k][i], L[temp_k][i]);
        }
        U[k][k] = A[k][k];
        for(int i=k+1; i<N; i++){
            L[i][k] = A[i][k]/U[k][k];
            U[k][i] = A[k][i];
        }
        for(int i=k+1; i<N; i++){
            for(int j=k+1; j<N; j++){
                A[i][j] -= L[i][k]*U[k][j];
            }
        }
    }

    for(int i=0; i<N; i++){
        int j = pi[i];
        P[i][j] = 1;
    }
}

void verifyLU(){
    vector<vector<double>> PA(N, vector<double>(N, 0.0));
    vector<vector<double>> LU(N, vector<double>(N, 0.0));
    vector<vector<double>> residual(N, vector<double>(N, 0.0));
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
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

    #ifdef DEBUG
    ofstream fout;
    fout.open(DEBUG_OUT_FILE);
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