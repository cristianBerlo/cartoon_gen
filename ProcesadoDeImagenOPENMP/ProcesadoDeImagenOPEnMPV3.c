#include <stdio.h>
#include <stdlib.h>
#include <FreeImage.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#pragma pack(push, 1)
typedef struct {
    BYTE rgbtBlue;
    BYTE rgbtGreen;
    BYTE rgbtRed;
} RGB;
#pragma pack(pop)

void filtroGausiano3(BYTE *img, BYTE *res, int pitch, int largo, int alto){
    const float kernel[3][3] = {
        { 1/16.0f, 2/16.0f, 1/16.0f },
        { 2/16.0f, 4/16.0f, 2/16.0f },
        { 1/16.0f, 2/16.0f, 1/16.0f }
    };
    #pragma omp parallel for schedule(static)
    for(int y = 1; y < alto - 1; y++){ 
        RGB* lineaRes = (RGB*)(res + (y * pitch));
        for(int x = 1; x < largo - 1; x++){
            float r=0, g=0, b=0;
            for(int oy = -1; oy <= 1; oy++){
                RGB* lineaImg = (RGB*)(img + ((y + oy) * pitch));
                for (int ox = -1; ox <= 1; ox++) {
                    r += (float)lineaImg[x + ox].rgbtRed   * kernel[1 + oy][1 + ox];
                    g += (float)lineaImg[x + ox].rgbtGreen * kernel[1 + oy][1 + ox];
                    b += (float)lineaImg[x + ox].rgbtBlue  * kernel[1 + oy][1 + ox];
                }
            }
            float gris = (0.2126f * r) + (0.7152f * g) + (0.0722f * b);
            if (gris < 0.0f) gris = 0.0f;
            if (gris > 255.0f) gris = 255.0f;
            lineaRes[x].rgbtRed   = (BYTE)gris;
            lineaRes[x].rgbtGreen = (BYTE)gris;
            lineaRes[x].rgbtBlue  = (BYTE)gris;
        }
    }
}

void filtroGausiano5(BYTE *img, BYTE *res, int pitch, int largo, int alto){
    const float kernel[5][5] = {
        { 0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f },
        { 0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f },
        { 0.023792f, 0.094907f, 0.150342f, 0.094907f, 0.023792f },
        { 0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f },
        { 0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f }
    };
    #pragma omp parallel for schedule(static)
    for(int y = 2; y < alto - 2; y++){ 
        RGB* lineaRes = (RGB*)(res + (y * pitch));
        for(int x = 2; x < largo - 2; x++){
            float r=0, g=0, b=0;
            for(int oy = -2; oy <= 2; oy++){
                RGB* lineaImg = (RGB*)(img + ((y + oy) * pitch));
                for (int ox = -2; ox <= 2; ox++) {
                    r += (float)lineaImg[x + ox].rgbtRed   * kernel[2 + oy][2 + ox];
                    g += (float)lineaImg[x + ox].rgbtGreen * kernel[2 + oy][2 + ox];
                    b += (float)lineaImg[x + ox].rgbtBlue  * kernel[2 + oy][2 + ox];
                }
            }
            float gris = (0.2126f * r) + (0.7152f * g) + (0.0722f * b);
            if (gris < 0.0f) gris = 0.0f;
            if (gris > 255.0f) gris = 255.0f;
            lineaRes[x].rgbtRed   = (BYTE)gris;
            lineaRes[x].rgbtGreen = (BYTE)gris;
            lineaRes[x].rgbtBlue  = (BYTE)gris;
        }
    }
}

void gradiente(BYTE *img, BYTE *res, int pitch, int largo, int alto){
    const int kernelX[3][3] = { {-1,0,1}, {-1,0,1}, {-1,0,1} };
    const int kernelY[3][3] = { {-1,-1,-1}, {0,0,0}, {1,1,1} };
	#pragma omp parallel for schedule(static)
    for(int y = 1; y < alto - 1; y++){ 
        RGB* lineaRes = (RGB*)(res + (y * pitch));
        for(int x = 1; x < largo - 1; x++){
            float rx=0, gx=0, bx=0;
            float ry=0, gy=0, by=0;
            for(int oy = -1; oy <= 1; oy++){
                RGB* lineaImg = (RGB*)(img + ((y + oy) * pitch));
                for (int ox = -1; ox <= 1; ox++) {
                    rx += lineaImg[x + ox].rgbtRed   * kernelX[1+oy][1+ox];
                    bx += lineaImg[x + ox].rgbtBlue  * kernelX[1+oy][1+ox];
                    gx += lineaImg[x + ox].rgbtGreen * kernelX[1+oy][1+ox];
                    ry += lineaImg[x + ox].rgbtRed   * kernelY[1+oy][1+ox];
                    by += lineaImg[x + ox].rgbtBlue  * kernelY[1+oy][1+ox];
                    gy += lineaImg[x + ox].rgbtGreen * kernelY[1+oy][1+ox];
                }
            }
            int resR = (int)sqrt((rx * rx) + (ry * ry));
            int resG = (int)sqrt((gx * gx) + (gy * gy));
            int resB = (int)sqrt((bx * bx) + (by * by));
            
            lineaRes[x].rgbtRed   = (BYTE)(resR > 255 ? 255 : resR);
            lineaRes[x].rgbtGreen = (BYTE)(resG > 255 ? 255 : resG);
            lineaRes[x].rgbtBlue  = (BYTE)(resB > 255 ? 255 : resB);
        }
    }
}

void umbralize(BYTE *res, int u, int pitch, int largo, int alto){
	#pragma omp parallel for schedule(static)
    for (int y = 0; y < alto; y++) {
        RGB* linea = (RGB*)(res + (y * pitch));
        for (int x = 0; x < largo; x++) {
            float gris = (0.2126f * linea[x].rgbtRed) +
                         (0.7152f * linea[x].rgbtGreen) +
                         (0.0722f * linea[x].rgbtBlue);
            BYTE binario = (gris >= u) ? 255 : 0;
            linea[x].rgbtRed   = binario;
            linea[x].rgbtGreen = binario;
            linea[x].rgbtBlue  = binario;
        }
    }
}

void posterize3(BYTE *res, int pitch, int largo, int alto){
	#pragma omp parallel for schedule(static)
    for (int y = 0; y < alto; y++) {
        RGB* linea = (RGB*)(res + (y * pitch));
        for (int x = 0; x < largo; x++) {
            linea[x].rgbtRed   = (linea[x].rgbtRed   <= 85) ? 0 : (linea[x].rgbtRed   <= 170) ? 127 : 255;
            linea[x].rgbtGreen = (linea[x].rgbtGreen <= 85) ? 0 : (linea[x].rgbtGreen <= 170) ? 127 : 255;
            linea[x].rgbtBlue  = (linea[x].rgbtBlue  <= 85) ? 0 : (linea[x].rgbtBlue  <= 170) ? 127 : 255;
        }
    }
}

void posterize9(BYTE *res, int pitch, int largo, int alto){
	#pragma omp parallel for schedule(static)
    for (int y = 0; y < alto; y++) {
        RGB* linea = (RGB*)(res + (y * pitch));
        for (int x = 0; x < largo; x++) {
            for(int c=0; c<3; c++) {
                BYTE *p = (c==0) ? &linea[x].rgbtRed : (c==1) ? &linea[x].rgbtGreen : &linea[x].rgbtBlue;
                *p = (*p <= 28)  ? 0   : (*p <= 56)  ? 32  : (*p <= 85)  ? 64  :
                     (*p <= 113) ? 96  : (*p <= 141) ? 128 : (*p <= 170) ? 159 :
                     (*p <= 198) ? 191 : (*p <= 226) ? 223 : 255;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    FreeImage_Initialise(FALSE);    
    FIBITMAP* bitmapO = NULL;
    FIBITMAP* bitmap = NULL; 
    FIBITMAP* bitmapRes = NULL;
    FIBITMAP* bitmapPost = NULL;

    double start, end;
    double cpu_time_used;

	omp_set_num_threads( atoi(argv[4]) );
    if (argc < 5) { 
        printf("Error: Faltan parámetros (Ruta, FiltroGauss[3|5], Posterize[3|9], numeroThreads).\n");
        FreeImage_DeInitialise();
        return 1;
    }

    const char* input_filename = argv[1];
    FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(input_filename, 0);
    if (fif == FIF_UNKNOWN) {
        fif = FreeImage_GetFIFFromFilename(input_filename);
    }

    if (fif == FIF_UNKNOWN) {
        printf("Error: Formato de imagen desconocido.\n");
        FreeImage_DeInitialise();
        return 1;
    }

    bitmapO = FreeImage_Load(fif, input_filename, 0);
    if (!bitmapO) {
        printf("Error: No se pudo cargar la imagen.\n");
        FreeImage_DeInitialise();
        return 1;
    }

    int ancho_orig = FreeImage_GetWidth(bitmapO);
    int alto_orig = FreeImage_GetHeight(bitmapO);
    int bpp_orig = FreeImage_GetBPP(bitmapO);
    
    int tam_borde = (atoi(argv[2]) == 5) ? 2 : 1;
    int nuevo_ancho = ancho_orig + (2 * tam_borde);
    int nuevo_alto = alto_orig + (2 * tam_borde);

    bitmap = FreeImage_Allocate(nuevo_ancho, nuevo_alto, bpp_orig, 0, 0, 0);
    if (!bitmap) {
        printf("Error: No se pudo crear el lienzo.\n");
        FreeImage_Unload(bitmapO);
        FreeImage_DeInitialise();
        return 1;
    }

    if (!FreeImage_Paste(bitmap, bitmapO, tam_borde, tam_borde, 255)) {
        printf("Error al aplicar el borde.\n");
    }
    FreeImage_Unload(bitmapO); 
    start = omp_get_wtime();
    int ancho = FreeImage_GetWidth(bitmap);
    int alto  = FreeImage_GetHeight(bitmap);
    int pitch = FreeImage_GetPitch(bitmap);

    bitmapRes = FreeImage_Clone(bitmap);
    bitmapPost = FreeImage_Clone(bitmap);

    BYTE* bits = FreeImage_GetBits(bitmap);
    BYTE* res = FreeImage_GetBits(bitmapRes);
    BYTE* post = FreeImage_GetBits(bitmapPost);

    int opc_gauss = atoi(argv[2]);
    if (opc_gauss == 5) {
        filtroGausiano5(bits, res, pitch, ancho, alto);
    } else {
        filtroGausiano3(bits, res, pitch, ancho, alto);
    }

    gradiente(res, bits, pitch, ancho, alto); 
    umbralize(bits, 80, pitch, ancho, alto); 

    int opc_post = atoi(argv[3]);
    if (opc_post == 9) {
        posterize9(post, pitch, ancho, alto);
    } else {
        posterize3(post, pitch, ancho, alto);
    }
    
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < alto; y++) {
        RGB* lineaPost = (RGB*)(post + (y * pitch));
        RGB* lineaBits = (RGB*)(bits + (y * pitch));
        for (int x = 0; x < ancho; x++) {
            if (lineaBits[x].rgbtRed == 255) {
                lineaPost[x].rgbtRed   = 0;
                lineaPost[x].rgbtGreen = 0;
                lineaPost[x].rgbtBlue  = 0;
            }
        }
    }

    end = omp_get_wtime();
    cpu_time_used = ((double) (end - start));
    printf("%f \n", cpu_time_used);
    FreeImage_Save(FIF_PNG, bitmapPost, "5.png", 0);

    FreeImage_Unload(bitmapRes);
    FreeImage_Unload(bitmapPost);
    FreeImage_Unload(bitmap);
    FreeImage_DeInitialise();
    return 0;
}
