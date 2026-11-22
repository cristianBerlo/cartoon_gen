#include <stdio.h>
#include <stdlib.h>
#include <FreeImage.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

#pragma pack(push, 1)
typedef struct {
    BYTE rgbtBlue;
    BYTE rgbtGreen;
    BYTE rgbtRed;
} RGB;
#pragma pack(pop)


void filtroGausiano3(BYTE *img, BYTE *res, int disp, int elemento, int pitch, int largo, int alto) {
    const float kernel[3][3] = {
        { 1/16.0f, 2/16.0f, 1/16.0f },
        { 2/16.0f, 4/16.0f, 2/16.0f },
        { 1/16.0f, 2/16.0f, 1/16.0f }
    };
    int filas = elemento / pitch;
    int filaInicio = disp / pitch;
    for (int y = 0; y < filas; y++) {
        int yGlobal = filaInicio + y;
        if (yGlobal == 0 || yGlobal >= alto - 1) continue;

        RGB* linea = (RGB*)(res + (y * pitch));

        for (int x = 1; x < largo - 1; x++) {
            float r = 0, g = 0, b = 0;
            for (int oy = -1; oy <= 1; oy++) {
                RGB* lineaO = (RGB*)(img + ((yGlobal + oy) * pitch));
                for (int ox = -1; ox <= 1; ox++) {
                    r += (float)lineaO[x + ox].rgbtRed   * kernel[1 + oy][1 + ox];
                    g += (float)lineaO[x + ox].rgbtGreen * kernel[1 + oy][1 + ox];
                    b += (float)lineaO[x + ox].rgbtBlue  * kernel[1 + oy][1 + ox];
                }
            }
            float gris = (0.2126f * r) + (0.7152f * g) + (0.0722f * b);
            if (gris < 0.0f) gris = 0.0f;
            if (gris > 255.0f) gris = 255.0f;
            linea[x].rgbtRed   = (BYTE)gris;
            linea[x].rgbtGreen = (BYTE)gris;
            linea[x].rgbtBlue  = (BYTE)gris;
        }
    }
}

void filtroGausiano5(BYTE *img, BYTE *res, int disp, int elemento, int pitch, int largo, int alto) {
	const float kernel[5][5] = {
		    { 0.003765, 0.015019, 0.023792, 0.015019, 0.003765 },
		    { 0.015019, 0.059912, 0.094907, 0.059912, 0.015019 },
		    { 0.023792, 0.094907, 0.150342, 0.094907, 0.023792 },
		    { 0.015019, 0.059912, 0.094907, 0.059912, 0.015019 },
		    { 0.003765, 0.015019, 0.023792, 0.015019, 0.003765 }
		};
    int filas = elemento / pitch;
    int filaInicio = disp / pitch;
    for (int y = 0; y < filas; y++) {
        int yGlobal = filaInicio + y;
    if (yGlobal < 2 || yGlobal >= alto - 2) continue;

        RGB* linea = (RGB*)(res + (y * pitch));

        for (int x = 2; x < largo - 2; x++) {
            float r = 0, g = 0, b = 0;
            for (int oy = -2; oy <= 2; oy++) {
                RGB* lineaO = (RGB*)(img + ((yGlobal + oy) * pitch));
                for (int ox = -2; ox <= 2; ox++) {
                    r += (float)lineaO[x + ox].rgbtRed   * kernel[2 + oy][2 + ox];
                    g += (float)lineaO[x + ox].rgbtGreen * kernel[2 + oy][2 + ox];
                    b += (float)lineaO[x + ox].rgbtBlue  * kernel[2 + oy][2 + ox];
                }
            }
            float gris = (0.2126f * r) + (0.7152f * g) + (0.0722f * b);
            if (gris < 0.0f) gris = 0.0f;
            if (gris > 255.0f) gris = 255.0f;
            linea[x].rgbtRed   = (BYTE)gris;
            linea[x].rgbtGreen = (BYTE)gris;
            linea[x].rgbtBlue  = (BYTE)gris;
        }
    }
}

void gradiente(BYTE *img, BYTE *res, int disp, int elemento, int pitch, int largo, int alto) {
	 const int kernelX[3][3]={
			 {-1,0,1},
			 {-1,0,1},
			 {-1,0,1}
	 };
	 const int kernelY[3][3]={
			 {-1,-1,-1},
			 {0,0,0},
			 {1,1,1}
	 };
    int filas = elemento / pitch;
    int filaInicio = disp / pitch;
    for (int y = 0; y < filas; y++) {
        int yGlobal = filaInicio + y;
        if (yGlobal == 0 || yGlobal >= alto - 1) continue;

        RGB* linea = (RGB*)(res + (y * pitch));

        for (int x = 1; x < largo - 1; x++) {
            float rx = 0, gx = 0, bx = 0;
            float ry=0, gy=0, by=0;
            for (int oy = -1; oy <= 1; oy++) {
                RGB* lineaO = (RGB*)(img + ((yGlobal + oy) * pitch));
                for (int ox = -1; ox <= 1; ox++) {
                	rx += lineaO[x + ox].rgbtRed *   kernelX[1+oy][1+ox];
                	bx += lineaO[x + ox].rgbtBlue *  kernelX[1+oy][1+ox];
                  	gx += lineaO[x + ox].rgbtGreen * kernelX[1+oy][1+ox];
                  	ry += lineaO[x + ox].rgbtRed *   kernelY[1+oy][1+ox];
                    by += lineaO[x + ox].rgbtBlue *  kernelY[1+oy][1+ox];
                    gy += lineaO[x + ox].rgbtGreen * kernelY[1+oy][1+ox];
                }
            }

        	int resR = (int)sqrt((rx * rx) + (ry * ry));
            int resG = (int)sqrt((gx * gx) + (gy * gy));
      		int resB = (int)sqrt((bx * bx) + (by * by));
            linea[x].rgbtRed   = (BYTE)resR;
            linea[x].rgbtGreen = (BYTE)resG;
            linea[x].rgbtBlue  = (BYTE)resB;
        }
    }
}

void umbralize( BYTE *res, int disp, int elemento, int pitch, int largo, int alto, int u) {
    int filas = elemento / pitch;
    for (int y = 0; y < filas; y++) {
        RGB* linea = (RGB*)(res + (y * pitch));

        for (int x = 1; x < largo - 1; x++) {
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

void posterize3(  BYTE *res, int disp, int elemento, int pitch, int largo, int alto) {
    int filas = elemento / pitch;
    int filaInicio = disp / pitch;

    for (int y = 0; y < filas; y++) {
        int yGlobal = filaInicio + y;
        RGB* linea = (RGB*)(res + (y * pitch));

        for (int x = 1; x < largo - 1; x++) {
        	linea[x].rgbtRed   = (linea[x].rgbtRed   <= 85) ? 0 : (linea[x].rgbtRed   <= 170) ? 127 : 255;
        	linea[x].rgbtGreen = (linea[x].rgbtGreen <= 85) ? 0 : (linea[x].rgbtGreen <= 170) ? 127 : 255;
            linea[x].rgbtBlue  = (linea[x].rgbtBlue  <= 85) ? 0 : (linea[x].rgbtBlue  <= 170) ? 127 : 255;
        }
    }
}

void posterize9( BYTE *res, int disp, int elemento, int pitch, int largo, int alto) {
    int filas = elemento / pitch;
    int filaInicio = disp / pitch;

    for (int y = 0; y < filas; y++) {
        int yGlobal = filaInicio + y;
        RGB* linea = (RGB*)(res + (y * pitch));

        for (int x = 1; x < largo - 1; x++) {
        	linea[x].rgbtRed = (linea[x].rgbtRed <= 28)  ? 0   :
        	        	                     (linea[x].rgbtRed <= 56)  ? 32  :
        	        	                     (linea[x].rgbtRed <= 85)  ? 64  :
        	        	                     (linea[x].rgbtRed <= 113) ? 96  :
        	        	                     (linea[x].rgbtRed <= 141) ? 128 :
        	        	                     (linea[x].rgbtRed <= 170) ? 159 :
        	        	                     (linea[x].rgbtRed <= 198) ? 191 :
        	        	                     (linea[x].rgbtRed <= 226) ? 223 : 255;

        	linea[x].rgbtGreen = (linea[x].rgbtGreen <= 28)  ? 0   :
        	        	                     (linea[x].rgbtGreen <= 56)  ? 32  :
        	        	                     (linea[x].rgbtGreen <= 85)  ? 64  :
        	        	                     (linea[x].rgbtGreen <= 113) ? 96  :
        	        	                     (linea[x].rgbtGreen <= 141) ? 128 :
        	        	                     (linea[x].rgbtGreen <= 170) ? 159 :
        	        	                     (linea[x].rgbtGreen <= 198) ? 191 :
        	        	                     (linea[x].rgbtGreen <= 226) ? 223 : 255;

            linea[x].rgbtBlue = (linea[x].rgbtBlue <= 28)  ? 0   :
        	        	                     (linea[x].rgbtBlue <= 56)  ? 32  :
        	        	                     (linea[x].rgbtBlue <= 85)  ? 64  :
        	        	                     (linea[x].rgbtBlue <= 113) ? 96  :
        	        	                     (linea[x].rgbtBlue <= 141) ? 128 :
        	        	                     (linea[x].rgbtBlue <= 170) ? 159 :
        	        	                     (linea[x].rgbtBlue <= 198) ? 191 :
        	        	                     (linea[x].rgbtBlue <= 226) ? 223 : 255;
        }
    }
}

int main(int argc, char *argv[]) {
    int node, size;
    int ancho = 0, alto = 0, bpp = 0, pitch = 0, elemento = 0, disp = 0;
    FIBITMAP* bitmapO = NULL;
    FIBITMAP* bitmap = NULL; 
    BYTE* bits = NULL;

    FreeImage_Initialise(FALSE);
    MPI_Init(&argc, &argv);
    double inicio, fin;

    MPI_Comm_rank(MPI_COMM_WORLD, &node);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int* elementos = (int*)malloc(size * sizeof(int));
    int* dis = (int*)malloc(size * sizeof(int));

    if (node == 0) {
        if (argc < 4) { 
            printf("Error: Faltan parámetros (Ruta, FiltroGauss, Posterize).\n");
            free(elementos); free(dis);
            FreeImage_DeInitialise();
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        const char* input_filename = argv[1];
        FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(input_filename, 0);
        if (fif == FIF_UNKNOWN) {
            fif = FreeImage_GetFIFFromFilename(input_filename);
        }

        if (fif == FIF_UNKNOWN) {
            printf("Error: Formato de imagen desconocido.\n");
            free(elementos); free(dis);
            FreeImage_DeInitialise();
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        bitmapO = FreeImage_Load(fif, input_filename, 0);
        if (!bitmapO) {
            printf("Error: No se pudo cargar la imagen.\n");
            free(elementos); free(dis);
            FreeImage_DeInitialise();
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        int ancho_orig = FreeImage_GetWidth(bitmapO);
        int alto_orig = FreeImage_GetHeight(bitmapO);
        int bpp_orig = FreeImage_GetBPP(bitmapO);

        int tam_borde = 1;
          if(atoi(argv[2])==5){
               tam_borde=2;}
        int nuevo_ancho = ancho_orig + (2 * tam_borde);
        int nuevo_alto = alto_orig + (2 * tam_borde);
        

        bitmap = FreeImage_Allocate(nuevo_ancho, nuevo_alto, bpp_orig, 0, 0, 0);
        
        if (!bitmap) {
            printf("Error: No se pudo crear el lienzo para el borde.\n");
            FreeImage_Unload(bitmapO);
            free(elementos); free(dis);
            FreeImage_DeInitialise();
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }


        int pos_x = tam_borde;
        int pos_y = tam_borde;
        if (!FreeImage_Paste(bitmap, bitmapO, pos_x, pos_y, 255)) {
            printf("Error al aplicar el borde.\n");
        }
        

        FreeImage_Unload(bitmapO); 


        ancho = FreeImage_GetWidth(bitmap);
        alto  = FreeImage_GetHeight(bitmap);
        bpp   = FreeImage_GetBPP(bitmap);
        pitch = FreeImage_GetPitch(bitmap);


        int resto = alto % size;
        int tot = 0;
        for (int i = 0; i < size; i++) {
            int filas_asignadas = (resto > 0) ? (alto / size) + 1 : (alto / size);
            elementos[i] = filas_asignadas * pitch;
            dis[i] = tot;
            tot += elementos[i];
            resto--;
        }
    }
    inicio = MPI_Wtime();
    int opc_gauss = 3;
    int opc_post = 3;
    opc_gauss = atoi(argv[2]);
    opc_post = atoi(argv[3]);
    MPI_Bcast(&opc_gauss, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&opc_post, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(elementos, 1, MPI_INT, &elemento, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(dis,       1, MPI_INT, &disp,     1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ancho, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alto,  1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bpp,   1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pitch, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (node != 0) {
        bitmap = FreeImage_Allocate(ancho, alto, bpp, 0, 0, 0);
    }
    
    bits = FreeImage_GetBits(bitmap);
    unsigned int memory_size = pitch * alto;
    MPI_Bcast(bits, memory_size, MPI_BYTE, 0, MPI_COMM_WORLD);

    int bytes = elemento;
    BYTE* resultadoLocal = (BYTE*)malloc(bytes);
    BYTE* resultadoLocalPost = (BYTE*)malloc(bytes);
    FIBITMAP* bitmap_Res = NULL;
    BYTE* bits_Res = NULL;
    
    if (node == 0) {
        bitmap_Res = FreeImage_Allocate(ancho, alto, bpp, 0, 0, 0);
        bits_Res = FreeImage_GetBits(bitmap_Res);
    }
    MPI_Scatterv( bits, elementos, dis, MPI_BYTE, resultadoLocalPost,  bytes,   MPI_BYTE, 0, MPI_COMM_WORLD);
    switch(opc_gauss){
        case 3:  filtroGausiano3( bits, resultadoLocal, disp, elemento, pitch, ancho, alto); break;
        case 5:  filtroGausiano5( bits, resultadoLocal, disp, elemento, pitch, ancho, alto); break;
        default: filtroGausiano3( bits, resultadoLocal, disp, elemento, pitch, ancho, alto); break;
    }
    
    switch(opc_post){
        case 3:  posterize3( resultadoLocalPost, disp, elemento, pitch, ancho, alto); break;
        case 9:  posterize9( resultadoLocalPost, disp, elemento, pitch, ancho, alto); break;
        default: posterize3( resultadoLocalPost, disp, elemento, pitch, ancho, alto); break;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(resultadoLocal, bytes, MPI_BYTE, bits_Res, elementos, dis, MPI_BYTE, 0, MPI_COMM_WORLD);
    
    if (node == 0) {
        memcpy(bits, bits_Res, memory_size);
    }
    
    MPI_Bcast(bits, memory_size, MPI_BYTE, 0, MPI_COMM_WORLD); 
    
    gradiente(bits, resultadoLocal, disp, elemento, pitch, ancho, alto);
    umbralize(resultadoLocal, disp, elemento, pitch, ancho, alto, 80);
    
    int filas = elemento / pitch;
    for (int y = 0; y < filas; y++) {
        RGB* linea = (RGB*)(resultadoLocalPost + (y * pitch));
        RGB* lineaO = (RGB*)(resultadoLocal + (y * pitch));
        for (int x = 0; x < ancho; x++) {
            if (lineaO[x].rgbtRed == 255) {
                linea[x].rgbtRed   = 0;
                linea[x].rgbtGreen = 0;
                linea[x].rgbtBlue  = 0;
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(resultadoLocalPost, bytes, MPI_BYTE, bits_Res, elementos, dis, MPI_BYTE, 0, MPI_COMM_WORLD);
      fin = MPI_Wtime();
    if (node == 0) {
        printf("%f\n", fin - inicio);
    }
    
    if (node == 0) {
        if (FreeImage_Save(FIF_PNG, bitmap_Res, "resultado_gauss2.png", 0)) {
            printf("[Nodo 0] Exito.\n");
        } else {
            printf("[Nodo 0] Error al guardar.\n");
        }
        FreeImage_Unload(bitmap_Res);
    }
    
    free(resultadoLocal);
    free(resultadoLocalPost);
    free(elementos);
    free(dis);
    if (bitmap) {
        FreeImage_Unload(bitmap);
    }
    FreeImage_DeInitialise();
    
  
    
    MPI_Finalize();
    return 0;
}