#include <bits/stdc++.h>
#include <fstream>
#include <chrono>
#include <time.h>
#include "constants.h"
#include <omp.h>
using namespace std;

double A[N][N];
double PA[N][N];
double LU[N][N];
double residual[N][N];
double* temp_A[PTHREAD_COUNT];
double l[N], u[N];
double L[N][N], U[N][N];
int P[N][N];
int pi[N];
int chunk_size = int(ceil((1.0*N) / (1.0*PTHREAD_COUNT)));
int chunk_size_2d = chunk_size * N ; 

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
    
    for (int i = 0; i < PTHREAD_COUNT; i++)
    {
        temp_A[i] = (double *)malloc(chunk_size_2d * sizeof(double));
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            U[i][j] = A[i][j];
            L[i][j] = A[i][j];
        }
    }
    for (int id = 0; id < PTHREAD_COUNT; id++){
        int ind = 0;
        for(int base_ptr = 0; base_ptr < chunk_size; base_ptr += 1){
            int base = base_ptr * PTHREAD_COUNT;
            int offset = id;
            for(int col = 0; col < N; col++){
                temp_A[id][ind] = A[base + offset][col];
                ind += 1;
            }
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

int get_temp_x(int i){
    int t_id = i % PTHREAD_COUNT;
    return t_id;
}

int get_temp_y(int i, int j){
    int col_id = (i / PTHREAD_COUNT) * N + j;
    return col_id;  
}

void LUdecompose()
{
    ofstream fout;

    #ifdef TIMING
    fout.open(DEBUG_OUT_FILE, ios::app);
    double total_A_time = 0.0;
    #endif
    
    auto start_time = chrono::high_resolution_clock::now();
    for (int k = 0; k < N; k++)
    {
       
        double maxi = 0.0;
        int temp_k = k;
        
        for(int i = k; i < N; i++)
        {
            int x = get_temp_x(i);
            int y = get_temp_y(i, k);
            if (maxi < temp_A[x][y])
            {
                maxi = temp_A[x][y];
                temp_k = i;
            }
        }
        #ifdef DEBUG
        if (maxi == 0.0)
        {
            perror("Singular matrix");
        }
        #endif
        
        U[k][k] = temp_A[get_temp_x(temp_k)][get_temp_y(temp_k, k)];
        swap(pi[k], pi[temp_k]);

        for(int j = 0; j < N; j++)
        {
            swap(temp_A[get_temp_x(k)][get_temp_y(k, j)], temp_A[get_temp_x(temp_k)][get_temp_y(temp_k, j)]);
        }

        for (int i = 0; i < k; i++)
        {
            swap(L[k][i], L[temp_k][i]);
        }
        
        #pragma omp parallel for num_threads(PTHREAD_COUNT) schedule(static) if (N - k - 1 > 100)
        for (int i = k + 1; i < N; i++)     
        {
            L[i][k] = temp_A[get_temp_x(i)][get_temp_y(i,k)] / U[k][k];
            U[k][i] = temp_A[get_temp_x(k)][get_temp_y(k,i)];
            l[i] = L[i][k];
            u[i] = U[k][i]; 
        }

        #ifdef TIMING
        auto inner_start_time = chrono::high_resolution_clock::now();
        #endif

        #pragma omp parallel num_threads(PTHREAD_COUNT)
        {
            int tid = omp_get_thread_num();
            for(int element=0; element<chunk_size_2d; element++){
                int i = (element/N)*PTHREAD_COUNT + tid;
                int j = element%N;
                temp_A[tid][element] -= l[i]*u[j];
            }
        }

        #ifdef TIMING
        auto inner_end_time = chrono::high_resolution_clock::now();
        
        double inner_time_taken = chrono::duration_cast<chrono::nanoseconds>(inner_end_time - inner_start_time).count();
        total_A_time += inner_time_taken;
        
        fout << "Inner Parallel Time: " << inner_time_taken << " ns\n" << endl;
        #endif

        // #pragma omp parallel for num_threads(PTHREAD_COUNT) schedule(static) if (N - k - 1 > 100)
        // for (int i = k + 1; i < N; i++)
        // {
        //     // #pragma omp simd aligned(l, u: 32)
        //     for (int j = k + 1; j < N; j+=1)
        //     {
        //         temp_A[i][j] -= l[i]*u[j];

        //     }
        // }
    }

    #ifdef TIMING
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
    fout << "N=" << N << ", OpenMP,"<< " Threads=" <<PTHREAD_COUNT <<", " << time_taken << " ms" << endl;
    #ifdef TIMING
    fout << "Total A update time OMP: " << total_A_time << " ns" << endl;
    #endif
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