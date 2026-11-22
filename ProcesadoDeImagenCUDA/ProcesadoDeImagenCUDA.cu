
#include <stdio.h>
#include <stdlib.h>
#include <FreeImage.h>
#include <math.h>

#pragma pack(push, 1)
typedef struct {
    int ancho, alto, bpp, pitch, elemento, disp;
} Datos;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct {
    BYTE rgbtBlue;
    BYTE rgbtGreen;
    BYTE rgbtRed;
} RGB;
#pragma pack(pop)

__global__ void umbralize(BYTE *imagen, Datos *datos);
__global__ void posterize3(BYTE *imagen, Datos *datos);
__global__ void posterize9(BYTE *imagen, Datos *datos);
__global__ void filtroGausiano3(BYTE *imagen, BYTE *imagenRes, Datos *datos);
__global__ void filtroGausiano5(BYTE *imagen, BYTE *imagenRes, Datos *datos);
__global__ void gradiente(BYTE *imagen, BYTE *imagenRes, Datos *datos);
__global__ void salida(BYTE *imagen, BYTE *imagenRes, Datos *datos);

int main(void) {
    FIBITMAP* bitmap0 = NULL;
    FIBITMAP* bitmap = NULL;
    BYTE* bits = NULL;
    Datos datos;
    const char* input_filename = "emmaStoner2400x2400.jpg";

    FreeImage_Initialise(FALSE);

    FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(input_filename, 0);
    if (fif == FIF_UNKNOWN) {
        fif = FreeImage_GetFIFFromFilename(input_filename);
    }

    if (fif == FIF_UNKNOWN) {
        printf("Error: Formato de imagen desconocido.\n");
        FreeImage_DeInitialise();
        return 1;
    }

    bitmap0 = FreeImage_Load(fif, input_filename, 0);
    if (!bitmap0) {
        printf("Error: No se pudo cargar la imagen.\n");
        FreeImage_DeInitialise();
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
        printf("Error: No se pudo cargar la imagen.\n");
        FreeImage_DeInitialise();
        return 1;
    }
    datos.ancho = FreeImage_GetWidth(bitmap);
    datos.alto  = FreeImage_GetHeight(bitmap);
    datos.bpp   = FreeImage_GetBPP(bitmap);
    datos.pitch = FreeImage_GetPitch(bitmap);
    bits = FreeImage_GetBits(bitmap);

    BYTE *dev_imagenPost, *dev_imagenGauss, *dev_imagenGradiente;
    Datos *dev_b;
    unsigned int memory_size = datos.pitch * datos.alto;
    //No encontre forma de resolver el problema usando solo dos imagenes.
    // Cuando llegaba a calcular el gradiente necesitaba la imagen resultante de gauss sin alterar para calcular todos los pixeles
    //Si alteraba imagenPost me quedaba sin la original para hacer el posterizado y si altero imagenGauss los bits a los que ya calcule el gradiente
    // van a modificar a los siguientes
    cudaMalloc((void**)&dev_imagenGauss, memory_size * sizeof(BYTE));
    cudaMalloc((void**)&dev_imagenPost, memory_size * sizeof(BYTE));
    cudaMalloc((void**)&dev_imagenGradiente, memory_size * sizeof(BYTE));
    cudaMalloc((void**)&dev_b, sizeof(Datos));

    cudaMemcpy(dev_imagenPost, bits, memory_size * sizeof(BYTE), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, &datos, sizeof(Datos), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((datos.ancho + blockSize.x - 1) / blockSize.x,
                  (datos.alto + blockSize.y - 1) / blockSize.y);
    filtroGausiano3<<<gridSize, blockSize>>>(dev_imagenPost, dev_imagenGauss, dev_b);
        cudaDeviceSynchronize();
    gradiente<<<gridSize, blockSize>>>(dev_imagenGauss, dev_imagenGradiente, dev_b);
    umbralize<<<gridSize, blockSize>>>(dev_imagenGradiente, dev_b);
        cudaDeviceSynchronize();
    posterize3<<<gridSize, blockSize>>>(dev_imagenPost, dev_b);
    cudaDeviceSynchronize();
    salida<<<gridSize, blockSize>>>(dev_imagenPost, dev_imagenGradiente, dev_b);
    cudaDeviceSynchronize();

    cudaMemcpy(bits, dev_imagenPost, memory_size * sizeof(BYTE), cudaMemcpyDeviceToHost);

    if(FreeImage_Save(FIF_PNG, bitmap, "salida_posterizada.png", 0)) {
        printf("Imagen procesada con éxito. Revisa 'salida_posterizada.png' en tus archivos.\n");
    } else {
        printf("Error al salvar la imagen.\n");
    }
    cudaFree(dev_imagenGradiente);
    cudaFree(dev_imagenGauss);
    cudaFree(dev_imagenPost);
    cudaFree(dev_b);
    FreeImage_Unload(bitmap);
    FreeImage_DeInitialise();

    return 0;
}

__global__ void umbralize(BYTE *imagen, Datos *datos) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < datos->ancho && y < datos->alto) {
        RGB* linea = (RGB*)(imagen + (y * datos->pitch));
        BYTE binario = (linea[x].rgbtBlue >= 80) ? 255 : 0;
        linea[x].rgbtRed   = binario;
        linea[x].rgbtGreen = binario;
        linea[x].rgbtBlue  = binario;
    }
}

__global__ void posterize3(BYTE *imagen, Datos *datos) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < datos->ancho && y < datos->alto) {
        RGB* linea = (RGB*)(imagen + (y * datos->pitch));
        linea[x].rgbtRed   = (linea[x].rgbtRed   <= 85) ? 0 : (linea[x].rgbtRed   <= 170) ? 127 : 255;
        linea[x].rgbtGreen = (linea[x].rgbtGreen <= 85) ? 0 : (linea[x].rgbtGreen <= 170) ? 127 : 255;
        linea[x].rgbtBlue  = (linea[x].rgbtBlue  <= 85) ? 0 : (linea[x].rgbtBlue  <= 170) ? 127 : 255;
    }
}

__global__ void posterize9(BYTE *imagen, Datos *datos){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < datos->ancho && y < datos->alto) {
        RGB* linea = (RGB*)(imagen + (y * datos->pitch));
        linea[x].rgbtRed = (linea[x].rgbtRed <= 28)  ? 0   : (linea[x].rgbtRed <= 56)  ? 32  : (linea[x].rgbtRed <= 85)  ? 64  : (linea[x].rgbtRed <= 113) ? 96  : (linea[x].rgbtRed <= 141) ? 128 : (linea[x].rgbtRed <= 170) ? 159 : (linea[x].rgbtRed <= 198) ? 191 : (linea[x].rgbtRed <= 226) ? 223 : 255;
        linea[x].rgbtGreen = (linea[x].rgbtGreen <= 28)  ? 0   : (linea[x].rgbtGreen <= 56)  ? 32  : (linea[x].rgbtGreen <= 85)  ? 64  : (linea[x].rgbtGreen <= 113) ? 96  : (linea[x].rgbtGreen <= 141) ? 128 : (linea[x].rgbtGreen <= 170) ? 159 : (linea[x].rgbtGreen <= 198) ? 191 : (linea[x].rgbtGreen <= 226) ? 223 : 255;
        linea[x].rgbtBlue = (linea[x].rgbtBlue <= 28)  ? 0   : (linea[x].rgbtBlue <= 56)  ? 32  : (linea[x].rgbtBlue <= 85)  ? 64  : (linea[x].rgbtBlue <= 113) ? 96  : (linea[x].rgbtBlue <= 141) ? 128 : (linea[x].rgbtBlue <= 170) ? 159 : (linea[x].rgbtBlue <= 198) ? 191 : (linea[x].rgbtBlue <= 226) ? 223 : 255; // Corregido 2159 -> 159
    }
}

__global__ void gradiente(BYTE *imagen, BYTE *imagenRes, Datos *datos){
    const int kernelX[3][3] = { {-1,0,1}, {-1,0,1}, {-1,0,1} };
    const int kernelY[3][3] = { {-1,-1,-1}, {0,0,0}, {1,1,1} };

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < datos->ancho && y < datos->alto) {
        RGB* linea = (RGB*)(imagenRes + (y * datos->pitch));
        float rx = 0, gx = 0, bx = 0;
        float ry = 0, gy = 0, by = 0;

        for(int oy = -1; oy <= 1; oy++){
            int coordY = max(0, min(datos->alto - 1, y + oy));
            RGB* lineaO = (RGB*)(imagen + (coordY * datos->pitch));

            for(int ox = -1; ox <= 1; ox++){
                int coordX = max(0, min(datos->ancho - 1, x + ox));

                rx += lineaO[coordX].rgbtRed   * kernelX[1+oy][1+ox];
                bx += lineaO[coordX].rgbtBlue  * kernelX[1+oy][1+ox];
                gx += lineaO[coordX].rgbtGreen * kernelX[1+oy][1+ox];
                ry += lineaO[coordX].rgbtRed   * kernelY[1+oy][1+ox];
                by += lineaO[coordX].rgbtBlue  * kernelY[1+oy][1+ox];
                gy += lineaO[coordX].rgbtGreen * kernelY[1+oy][1+ox];
            }
        }
        int resR = (int)sqrt((rx * rx) + (ry * ry));
        int resG = (int)sqrt((gx * gx) + (gy * gy));
        int resB = (int)sqrt((bx * bx) + (by * by));

        linea[x].rgbtRed   = (BYTE)max(0, min(255, resR));
        linea[x].rgbtGreen = (BYTE)max(0, min(255, resG));
        linea[x].rgbtBlue  = (BYTE)max(0, min(255, resB));
    }
}

__global__ void filtroGausiano3(BYTE *imagen, BYTE *imagenRes, Datos *datos){
    const float kernel[3][3] = {
        { 1/16.0f, 2/16.0f, 1/16.0f },
        { 2/16.0f, 4/16.0f, 2/16.0f },
        { 1/16.0f, 2/16.0f, 1/16.0f }
    };
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < datos->ancho && y < datos->alto) {
        RGB* linea = (RGB*)(imagenRes + (y * datos->pitch));
        float r = 0, g = 0, b = 0;
        for(int oy = -1; oy <= 1; oy++){
            int coordY = max(0, min(datos->alto - 1, y + oy));
            RGB* lineaO = (RGB*)(imagen + (coordY * datos->pitch));
            for(int ox = -1; ox <= 1; ox++){
                int coordX = max(0, min(datos->ancho - 1, x + ox));
                r += (float)lineaO[coordX].rgbtRed   * kernel[1 + oy][1 + ox];
                g += (float)lineaO[coordX].rgbtGreen * kernel[1 + oy][1 + ox];
                b += (float)lineaO[coordX].rgbtBlue  * kernel[1 + oy][1 + ox];
            }
        }
        float gris = (0.2126f * r) + (0.7152f * g) + (0.0722f * b);
        gris = max(0.0f, min(255.0f, gris));

        linea[x].rgbtRed   = (BYTE)gris;
        linea[x].rgbtGreen = (BYTE)gris;
        linea[x].rgbtBlue  = (BYTE)gris;
    }
}

__global__ void filtroGausiano5(BYTE *imagen, BYTE *imagenRes, Datos *datos){
    const float kernel[5][5] = {
        { 0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f },
        { 0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f },
        { 0.023792f, 0.094907f, 0.150342f, 0.094907f, 0.023792f },
        { 0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f },
        { 0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f }
    };
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < datos->ancho && y < datos->alto) {
        RGB* linea = (RGB*)(imagenRes + (y * datos->pitch));
        float r = 0, g = 0, b = 0;
        for(int oy = -2; oy <= 2; oy++){
            int coordY = max(0, min(datos->alto - 1, y + oy));
            RGB* lineaO = (RGB*)(imagen + (coordY * datos->pitch));
            for(int ox = -2; ox <= 2; ox++){
                int coordX = max(0, min(datos->ancho - 1, x + ox));
                r += (float)lineaO[coordX].rgbtRed   * kernel[2 + oy][2 + ox];
                g += (float)lineaO[coordX].rgbtGreen * kernel[2 + oy][2 + ox];
                b += (float)lineaO[coordX].rgbtBlue  * kernel[2 + oy][2 + ox];
            }
        }
        float gris = (0.2126f * r) + (0.7152f * g) + (0.0722f * b);
        gris = max(0.0f, min(255.0f, gris));
        linea[x].rgbtRed   = (BYTE)gris;
        linea[x].rgbtGreen = (BYTE)gris;
        linea[x].rgbtBlue  = (BYTE)gris;
    }
}

__global__ void salida(BYTE *imagen, BYTE *imagenRes, Datos *datos){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < datos->ancho && y < datos->alto) {
        RGB* linea = (RGB*)(imagen + (y * datos->pitch));
        RGB* lineaO = (RGB*)(imagenRes + (y * datos->pitch));
        if (lineaO[x].rgbtRed == 255) {
            linea[x].rgbtRed   = 0;
            linea[x].rgbtGreen = 0;
            linea[x].rgbtBlue  = 0;
        }
    }
}
