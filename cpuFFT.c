#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

typedef struct {
    float real, imag;
} cmplx;

cmplx cmplx_mul_add(const cmplx c, const cmplx a, const cmplx b) {
    const cmplx ret = {
            (a.real * b.real) + c.real - (a.imag * b.imag),
            (a.imag * b.real) + (a.real * b.imag) + c.imag
    };
    return ret;
}

void fft_Stockham(const cmplx *input, cmplx *output, int n, int flag) {
    int half = n >> 1;
    cmplx *buffer = (cmplx *) calloc(sizeof(cmplx), n * 2);
    if (buffer == NULL)
        return;
    cmplx *tmp = buffer;
    cmplx *y = tmp + n;
    memcpy(y, input, sizeof(cmplx) * n);
    for (int r = half, l = 1; r >= 1; r >>= 1) {
        cmplx *tp = y;
        y = tmp;
        tmp = tp;
        float factor_w = -flag * M_PI / l;
        cmplx w = {cosf(factor_w), sinf(factor_w)};
        cmplx wj = {1, 0};
        for (int j = 0; j < l; j++) {
            int jrs = j * (r << 1);
            for (int k = jrs, m = jrs >> 1; k < jrs + r; k++) {
                const cmplx t = {(wj.real * tmp[k + r].real) - (wj.imag * tmp[k + r].imag),
                                 (wj.imag * tmp[k + r].real) + (wj.real * tmp[k + r].imag)};
                y[m].real = tmp[k].real + t.real;
                y[m].imag = tmp[k].imag + t.imag;
                y[m + half].real = tmp[k].real - t.real;
                y[m + half].imag = tmp[k].imag - t.imag;
                m++;
            }
            const float t = wj.real;
            wj.real = (t * w.real) - (wj.imag * w.imag);
            wj.imag = (wj.imag * w.real) + (t * w.imag);
        }
        l <<= 1;
    }
    memcpy(output, y, sizeof(cmplx) * n);
    free(buffer);
}

void fft_radix3(const cmplx *input, cmplx *output, int n, int flag) {
    if (n < 2) {
        output[0] = input[0];
        return;
    }
    int radix = 3;
    int np = n / radix;
    cmplx *res = (cmplx *) malloc(sizeof(cmplx) * n);
    cmplx *f0 = res;
    cmplx *f1 = f0 + np;
    cmplx *f2 = f1 + np;
    for (int i = 0; i < np; i++) {
        for (int j = 0; j < radix; j++) {
            res[i + j * np] = input[radix * i + j];
        }
    }
    fft_radix3(f0, f0, np, flag);
    fft_radix3(f1, f1, np, flag);
    fft_radix3(f2, f2, np, flag);
    float wexp0 = -2 * M_PI * flag / n;
    cmplx wt = {cosf(wexp0), sinf(wexp0)};
    cmplx w0 = {1, 0};
    for (int i = 0; i < np; i++) {
        const float w0r = w0.real;
        w0.real = (w0r * wt.real) - (w0.imag * wt.imag);
        w0.imag = (w0.imag * wt.real) + (w0r * wt.imag);
    }
    cmplx w = {1, 0};
    for (int j = 0; j < radix; j++) {
        cmplx wj = w;
        for (int k = 0; k < np; k++) {
            output[k + j * np] = cmplx_mul_add(f0[k], cmplx_mul_add(f1[k], f2[k], wj), wj);
            const float wjr = wj.real;
            wj.real = (wjr * wt.real) - (wj.imag * wt.imag);
            wj.imag = (wj.imag * wt.real) + (wjr * wt.imag);
        }
        const float wr = w.real;
        w.real = (wr * w0.real) - (w.imag * w0.imag);
        w.imag = (w.imag * w0.real) + (wr * w0.imag);
    }
    free(res);
}

void fft_radix5(const cmplx *input, cmplx *output, int n, int flag) {
    if (n < 2) {
        output[0] = input[0];
        return;
    }
    int radix = 5;
    int np = n / radix;
    cmplx *res = (cmplx *) calloc(sizeof(cmplx), n);
    cmplx *f0 = res;
    cmplx *f1 = f0 + np;
    cmplx *f2 = f1 + np;
    cmplx *f3 = f2 + np;
    cmplx *f4 = f3 + np;
    for (int i = 0; i < np; i++) {
        for (int j = 0; j < radix; j++) {
            res[i + j * np] = input[radix * i + j];
        }
    }
    fft_radix5(f0, f0, np, flag);
    fft_radix5(f1, f1, np, flag);
    fft_radix5(f2, f2, np, flag);
    fft_radix5(f3, f3, np, flag);
    fft_radix5(f4, f4, np, flag);
    float wexp0 = -2 * M_PI * flag / n;
    cmplx wt = {cosf(wexp0), sinf(wexp0)};
    cmplx w0 = {1, 0};
    for (int i = 0; i < np; i++) {
        const float w0r = w0.real;
        w0.real = (w0r * wt.real) - (w0.imag * wt.imag);
        w0.imag = (w0.imag * wt.real) + (w0r * wt.imag);
    }
    cmplx w = {1, 0};
    for (int j = 0; j < radix; j++) {
        cmplx wj = w;
        for (int k = 0; k < np; k++) {
            output[k + j * np] = cmplx_mul_add(f0[k], cmplx_mul_add(f1[k], cmplx_mul_add(f2[k],
                                                                                         cmplx_mul_add(f3[k], f4[k],
                                                                                                       wj), wj), wj),
                                               wj);
            const float wjr = wj.real;
            wj.real = (wjr * wt.real) - (wj.imag * wt.imag);
            wj.imag = (wj.imag * wt.real) + (wjr * wt.imag);
        }
        const float wr = w.real;
        w.real = (wr * w0.real) - (w.imag * w0.imag);
        w.imag = (w.imag * w0.real) + (wr * w0.imag);
    }
    free(res);
}

void fft_radix6(const cmplx *input, cmplx *output, int n, int flag) {
    if (n < 2) {
        output[0] = input[0];
        return;
    }
    int radix = 6;
    int np = n / radix;
    cmplx *res = (cmplx *) calloc(sizeof(cmplx), n);
    cmplx *f0 = res;
    cmplx *f1 = f0 + np;
    cmplx *f2 = f1 + np;
    cmplx *f3 = f2 + np;
    cmplx *f4 = f3 + np;
    cmplx *f5 = f4 + np;
    for (int i = 0; i < np; i++) {
        for (int j = 0; j < radix; j++) {
            res[i + j * np] = input[radix * i + j];
        }
    }
    fft_radix6(f0, f0, np, flag);
    fft_radix6(f1, f1, np, flag);
    fft_radix6(f2, f2, np, flag);
    fft_radix6(f3, f3, np, flag);
    fft_radix6(f4, f4, np, flag);
    fft_radix6(f5, f5, np, flag);
    float wexp0 = -2 * M_PI * flag / n;
    cmplx wt = {cosf(wexp0), sinf(wexp0)};
    cmplx w0 = {1, 0};
    for (int i = 0; i < np; i++) {
        const float w0r = w0.real;
        w0.real = (w0r * wt.real) - (w0.imag * wt.imag);
        w0.imag = (w0.imag * wt.real) + (w0r * wt.imag);
    }
    cmplx w = {1, 0};
    for (int j = 0; j < radix; j++) {
        cmplx wj = w;
        for (int k = 0; k < np; k++) {
            output[k + j * np] = cmplx_mul_add(f0[k], cmplx_mul_add(f1[k], cmplx_mul_add(f2[k],
                                                                                         cmplx_mul_add(f3[k],
                                                                                                       cmplx_mul_add(
                                                                                                               f4[k],
                                                                                                               f5[k],
                                                                                                               wj), wj),
                                                                                         wj), wj), wj);
            const float wjr = wj.real;
            wj.real = (wjr * wt.real) - (wj.imag * wt.imag);
            wj.imag = (wj.imag * wt.real) + (wjr * wt.imag);
        }
        const float wr = w.real;
        w.real = (wr * w0.real) - (w.imag * w0.imag);
        w.imag = (w.imag * w0.real) + (wr * w0.imag);
    }
    free(res);
}

void fft_radix7(const cmplx *input, cmplx *output, int n, int flag) {
    if (n < 2) {
        output[0] = input[0];
        return;
    }
    int radix = 7;
    int np = n / radix;
    cmplx *res = (cmplx *) calloc(sizeof(cmplx), n);
    cmplx *f0 = res;
    cmplx *f1 = f0 + np;
    cmplx *f2 = f1 + np;
    cmplx *f3 = f2 + np;
    cmplx *f4 = f3 + np;
    cmplx *f5 = f4 + np;
    cmplx *f6 = f5 + np;
    for (int i = 0; i < np; i++) {
        for (int j = 0; j < radix; j++) {
            res[i + j * np] = input[radix * i + j];
        }
    }
    fft_radix7(f0, f0, np, flag);
    fft_radix7(f1, f1, np, flag);
    fft_radix7(f2, f2, np, flag);
    fft_radix7(f3, f3, np, flag);
    fft_radix7(f4, f4, np, flag);
    fft_radix7(f5, f5, np, flag);
    fft_radix7(f6, f6, np, flag);
    float wexp0 = -2 * M_PI * flag / n;
    cmplx wt = {cosf(wexp0), sinf(wexp0)};
    cmplx w0 = {1, 0};
    for (int i = 0; i < np; i++) {
        const float w0r = w0.real;
        w0.real = (w0r * wt.real) - (w0.imag * wt.imag);
        w0.imag = (w0.imag * wt.real) + (w0r * wt.imag);
    }
    cmplx w = {1, 0};
    for (int j = 0; j < radix; j++) {
        cmplx wj = w;
        for (int k = 0; k < np; k++) {
            output[k + j * np] = cmplx_mul_add(f0[k], cmplx_mul_add(f1[k], cmplx_mul_add(f2[k],
                                                                                         cmplx_mul_add(f3[k],
                                                                                                       cmplx_mul_add(
                                                                                                               f4[k],
                                                                                                               cmplx_mul_add(
                                                                                                                       f5[k],
                                                                                                                       f6[k],
                                                                                                                       wj),
                                                                                                               wj), wj),
                                                                                         wj), wj), wj);
            const float wjr = wj.real;
            wj.real = (wjr * wt.real) - (wj.imag * wt.imag);
            wj.imag = (wj.imag * wt.real) + (wjr * wt.imag);
        }
        const float wr = w.real;
        w.real = (wr * w0.real) - (w.imag * w0.imag);
        w.imag = (w.imag * w0.real) + (wr * w0.imag);
    }
    free(res);
}

void fft_Bluestein(const cmplx *input, cmplx *output, int n, int flag) {
    int m = 1 << ((unsigned int) (ilogbf((float) (2 * n - 1))));
    if (m < 2 * n - 1) {
        m <<= 1;
    }
    cmplx *y = (cmplx *) calloc(sizeof(cmplx), 3 * m);
    if (y == NULL)
        return;
    cmplx *w = y + m;
    cmplx *ww = w + m;
    w[0].real = 1;
    if (flag == -1) {
        y[0].real = input[0].real;
        y[0].imag = -input[0].imag;
        for (int i = 1; i < n; i++) {
            const float wexp = M_PI * i * i / n;
            w[i].real = cosf(wexp);
            w[i].imag = sinf(wexp);
            w[m - i] = w[i];
            y[i].real = (input[i].real * w[i].real) - (input[i].imag * w[i].imag);
            y[i].imag = (-input[i].imag * w[i].real) - (input[i].real * w[i].imag);
        }
    } else {
        y[0].real = input[0].real;
        y[0].imag = input[0].imag;
        for (int i = 1; i < n; i++) {
            const float wexp = M_PI * i * i / n;
            w[i].real = cosf(wexp);
            w[i].imag = sinf(wexp);
            w[m - i] = w[i];
            y[i].real = (input[i].real * w[i].real) + (input[i].imag * w[i].imag);
            y[i].imag = (input[i].imag * w[i].real) - (input[i].real * w[i].imag);
        }
    }
    fft_Stockham(y, y, m, 1);
    fft_Stockham(w, ww, m, 1);
    for (int i = 0; i < m; i++) {
        const float r = y[i].real;
        y[i].real = (r * ww[i].real) - (y[i].imag * ww[i].imag);
        y[i].imag = (y[i].imag * ww[i].real) + (r * ww[i].imag);
    }
    fft_Stockham(y, y, m, -1);
    if (flag == -1) {
        for (int i = 0; i < n; i++) {
            output[i].real = ((y[i].real * w[i].real) + (y[i].imag * w[i].imag)) / m;
            output[i].imag = -((y[i].imag * w[i].real) - (y[i].real * w[i].imag)) / m;
        }
    } else {
        for (int i = 0; i < n; i++) {
            output[i].real = ((y[i].real * w[i].real) + (y[i].imag * w[i].imag)) / m;
            output[i].imag = ((y[i].imag * w[i].real) - (y[i].real * w[i].imag)) / m;
        }
    }
    free(y);
}

int base(int n) {
    int t = n & (n - 1);
    if (t == 0) {
        return 2;
    }
    for (int i = 3; i <= 7; i++) {
        int n2 = n;
        while (n2 % i == 0) {
            n2 /= i;
        }
        if (n2 == 1) {
            return i;
        }
    }
    return n;
}

void FFT(const cmplx *input, cmplx *output, int n) {
    if (n < 2) {
        output[0] = input[0];
        return;
    }
    int p = base(n);
    switch (p) {
        case 2:
            fft_Stockham(input, output, n, 1);
            break;
        case 3:
            fft_radix3(input, output, n, 1);
            break;
        case 5:
            fft_radix5(input, output, n, 1);
            break;
        case 6:
            fft_radix6(input, output, n, 1);
            break;
        case 7:
            fft_radix7(input, output, n, 1);
            break;
        default:
            fft_Bluestein(input, output, n, 1);
            break;
    }
}

void IFFT(const cmplx *input, cmplx *output, int n) {
    if (n < 2) {
        output[0] = input[0];
        return;
    }
    int p = base(n);
    switch (p) {
        case 2:
            fft_Stockham(input, output, n, -1);
            break;
        case 3:
            fft_radix3(input, output, n, -1);
            break;
        case 5:
            fft_radix5(input, output, n, -1);
            break;
        case 6:
            fft_radix6(input, output, n, -1);
            break;
        case 7:
            fft_radix7(input, output, n, -1);
            break;
        default: {
            fft_Bluestein(input, output, n, -1);
            break;
        }
    }
    for (int i = 0; i < n; i++) {
        output[i].real = output[i].real / n;
        output[i].imag = output[i].imag / n;
    }
}

int main() {
    printf("Fast Fourier Transform\n");
    printf("blog: http://cpuimage.cnblogs.com/\n");
    printf("A Simple and Efficient FFT Implementation in C");
    size_t N = 513;
    cmplx *input = (cmplx *) calloc(sizeof(cmplx), N);
    cmplx *output = (cmplx *) calloc(sizeof(cmplx), N);
    for (int i = 0; i < N; ++i) {
        input[i].real = i;
        input[i].imag = 0;
    }
    for (int i = 0; i < N; ++i) {
        printf("(%f %f) \t", input[i].real, input[i].imag);
    }
    for (int i = 0; i < 100; i++) {
        FFT(input, output, N);
    }
    printf("\n");
    IFFT(output, input, N);
    for (int i = 0; i < N; ++i) {
        printf("(%f %f) \t", input[i].real, input[i].imag);
    }
    free(input);
    free(output);
    getchar();
    return 0;
}