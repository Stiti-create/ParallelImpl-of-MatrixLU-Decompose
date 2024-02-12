#include <bits/stdc++.h>
#include <fstream>
#include <chrono>
#include <time.h>
#include "constants.h"
#include <omp.h>
using namespace std;

// vector<int> pi(N, 0);
vector<vector<double>> A(N, vector<double>(N, 0.0));
// vector<vector<double>> L(N, vector<double>(N, 0.0));
// vector<vector<double>> U(N, vector<double>(N, 0.0));
// vector<vector<int>> P(N, vector<int>(N, 0));
vector<vector<double>> PA(N, vector<double>(N, 0.0));
vector<vector<double>> LU(N, vector<double>(N, 0.0));
vector<vector<double>> residual(N, vector<double>(N, 0.0));
// double temp_A[N][N];
// vector<vector<double>> temp_A(N, vector<double>(N, 0.0));
double temp_A[N][N];
double l[N], u[N];
double L[N][N], U[N][N];
int P[N][N];
int pi[N];


void inputMatrix()
{
    ifstream fin;
    fin.open(INPUT_MATRIX_FILE);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            fin >> A[i][j];
            if (fin.eof())
                break;
        }
    }
    fin.close();
}

void initOutputs()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            U[i][j] = A[i][j];
            L[i][j] = A[i][j];
            temp_A[i][j] = A[i][j];
        }
    }
    for (int i = 0; i < N; i++)
    {
        pi[i] = i;
        L[i][i] = 1.0;
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (i > j)
            {
                U[i][j] = 0;
            }
            if (i < j)
            {
                L[i][j] = 0;
            }
        }
    }
}

void LUdecompose()
{
    ofstream fout;
    // fout.open(DEBUG_OUT_FILE, ios::app);
    double total_A_time = 0.0;
    auto start_time = chrono::high_resolution_clock::now();
    for (int k = 0; k < N; k++)
    {
       
        double maxi = 0.0;
        int temp_k = k;
        
        for(int i = k; i < N; i++)
        {
            if (maxi < temp_A[i][k])
            {
                maxi = temp_A[i][k];
                temp_k = i;
            }
        }

        if (maxi == 0.0)
        {
            perror("Singular matrix");
        }
        
        U[k][k] = temp_A[temp_k][k];
        swap(pi[k], pi[temp_k]);
        swap(temp_A[k], temp_A[temp_k]);
        // THIS MIGHT CREATE OVERHEAD
        // #pragma omp parallel for num_threads(PTHREAD_COUNT) schedule(static)
        for (int i = 0; i < k; i++)
        {
            swap(L[k][i], L[temp_k][i]);
        }
        
        #pragma omp parallel for num_threads(PTHREAD_COUNT) schedule(static) if (N - k - 1 > 100)
        for (int i = k + 1; i < N; i++)     
        {
            L[i][k] = temp_A[i][k] / U[k][k];
            U[k][i] = temp_A[k][i];
            l[i] = L[i][k];
            u[i] = U[k][i]; 
        }
        // auto inner_start_time = chrono::high_resolution_clock::now();
        #pragma omp parallel for num_threads(PTHREAD_COUNT) schedule(static) if (N - k - 1 > 100)
        for (int i = k + 1; i < N; i++)
        {
            // #pragma omp simd aligned(l, u: 32)
            for (int j = k + 1; j < N; j++)
            {
                temp_A[i][j] -= l[i]*u[j];
            }
        }
        #ifdef DEBUG
        auto inner_end_time = chrono::high_resolution_clock::now();
        
        double inner_time_taken = chrono::duration_cast<chrono::nanoseconds>(inner_end_time - inner_start_time).count();
        total_A_time += inner_time_taken;
        
        fout << "Inner Parallel Time: " << inner_time_taken << " ns\n" << endl;
        #endif
    }

    #ifdef DEBUG
        fout << "Total A Time: " << total_A_time << " ns" << endl;
        fout.close();
    #endif

    #pragma omp parallel for num_threads(PTHREAD_COUNT)
    for (int i = 0; i < N; i++)
    {
        P[i][pi[i]] = 1;
    }
    
    auto end_time = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    fout.open(LOG_OUT_FILE, ios::app);
    fout << "-----------------------------------------------\n";
    fout << "N: " << N << ", Parallel(OpenMP): " << time_taken << " ms" << endl;
    // fout << "Total swap time seq: " << total_A_time << " ns" << endl;
    fout.close();
    return;
}

void verifyLU()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            PA[i][j] = 0.0;
            LU[i][j] = 0.0;
            for (int k = 0; k < N; k++)
            {
                PA[i][j] += P[i][k] * A[k][j];
                LU[i][j] += L[i][k] * U[k][j];
            }
        }
    }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            residual[i][j] = PA[i][j] - LU[i][j];
        }
    }

#ifdef DEBUG_LU_VERIFY
    ofstream fout;
    fout.open(LU_VERIFY_OUT, ios::app);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            fout << residual[i][j] << " ";
        }
        fout << endl;
    }
    fout.close();
#endif

    double norm = 0.0;
    for (int j = 0; j < N; j++)
    {
        double col_eucledian_norm = 0.0;
        for (int i = 0; i < N; i++)
        {
            col_eucledian_norm += pow(residual[i][j], 2);
        }
        norm += sqrt(col_eucledian_norm);
    }
    cout << "L-2,1 Norm of residual: " << norm << endl;
}

int main()
{
    inputMatrix();
    initOutputs();
    LUdecompose();
    // verifyLU();
}