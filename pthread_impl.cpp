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
vector<double *> temp_A_ptrs(N, NULL);
pthread_mutex_t lock_1;
pthread_mutex_t lock_2;
pthread_barrier_t barrier_1;

struct pthread_args
{
    int k;
    int thread_id;
    int temp_k;
    int l_2, r_2, l_3, r_3;
    vector<double *> local_A;
};

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
        temp_A_ptrs[i] = &temp_A[i][0];
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

pair<int, int> getBounds(int id, int num_threads, int num_iter)
{
    int q = num_iter / num_threads;
    int c = int(ceil(num_iter / (1.0 * num_threads)));
    int r = num_iter % num_threads;
    int base = r * c;
    if (id < r)
    {
        return {id * c, (id + 1) * c};
    }
    else
    {
        return {base + (id - r) * q, base + (id + 1 - r) * q};
    }
}

void *parallel_swap_LU(void *pthread_args)
{
    struct pthread_args *args = (struct pthread_args *)pthread_args;
    int k = args->k;
    int temp_k = args->temp_k;
    int id = args->thread_id;
    for (int j = args->l_2; j < args->r_2; j++)
    {
        swap(L[k][j], L[temp_k][j]);
    }
#ifdef DEBUG
    ofstream fout;
    fout.open(DEBUG_OUT_FILE, ios::app);
    fout << "Thread " << id << " k: " << k << " l_2: " << l_2 << " r_2: " << r_2 << endl;
#endif

    for (int ind = args->l_3; ind < args->r_3; ind++)
    {
        L[ind][k] = (temp_A_ptrs[ind][k] * 1.0) / U[k][k];
        U[k][ind] = temp_A_ptrs[k][ind];
    }
    vector<double *> local_A = args->local_A;

    pthread_barrier_wait(&barrier_1);

    for (int i = args->l_3; i < args->r_3; i++)
    {
        for (int j = k + 1; j < N; j++)
        {
            local_A[i - args->l_3][j] -= L[i][k] * U[k][j];
        }
    }

    pthread_exit(NULL);
}

void LUdecompose()
{
    pthread_args args[PTHREAD_COUNT];
    pthread_t threads[PTHREAD_COUNT];
    auto start_time = chrono::high_resolution_clock::now();
    for (int k = 0; k < N; k++)
    {
        double maxi = 0.0;
        int temp_k = k;
        for (int i = k; i < N; i++)
        {
            if (maxi < temp_A_ptrs[i][k])
            {
                maxi = temp_A_ptrs[i][k];
                temp_k = i;
            }
        }
        if (maxi == 0.0)
        {
            perror("Singular matrix");
        }
        U[k][k] = temp_A_ptrs[temp_k][k];
        swap(pi[k], pi[temp_k]);
        swap(temp_A_ptrs[k], temp_A_ptrs[temp_k]);

        pthread_barrier_init(&barrier_1, NULL, PTHREAD_COUNT);

        for (int num = 0; num < PTHREAD_COUNT; num++)
        {
            args[num].k = k;
            args[num].thread_id = num;
            args[num].temp_k = temp_k;
            pair<int, int> bounds_2 = getBounds(num, PTHREAD_COUNT, k);
            args[num].l_2 = bounds_2.first;
            args[num].r_2 = bounds_2.second;
            pair<int, int> bounds = getBounds(num, PTHREAD_COUNT, N - k - 1);
            args[num].l_3 = k + 1 + bounds.first;
            args[num].r_3 = k + 1 + bounds.second;
            // local to each thread vector to pointers to l-r rows of temp_A
            vector<double *> local_A_thread;
            copy(temp_A_ptrs.begin() + args[num].l_3, temp_A_ptrs.begin() + args[num].r_3, back_inserter(local_A_thread));
            args[num].local_A = local_A_thread;
            pthread_create(&threads[num], NULL, &parallel_swap_LU, (void *)&args[num]);
        }

        for (int num = 0; num < PTHREAD_COUNT; num++)
        {
            pair<int, int> bounds = getBounds(num, PTHREAD_COUNT, N - k - 1);
            int l = k + 1 + bounds.first;
            // Efficiently write back from local A to temp_A
            copy(args[num].local_A.begin(), args[num].local_A.end(), temp_A_ptrs.begin() + l);
            pthread_join(threads[num], NULL);
        }
    }

    // can be done parallelly in future, let it be sequential for now
    for (int i = 0; i < N; i++)
    {
        P[i][pi[i]] = 1;
    }
    auto end_time = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    ofstream fout;
    fout.open(LOG_OUT_FILE, ios::app);
    fout << "-----------------------------------------------\n";
    fout << "thread,"<<PTHREAD_COUNT<<",N," << N << ",pthread," << time_taken << ",ms" << endl;
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
    fout.open(LU_VERIFY_OUT);
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
    verifyLU();
}