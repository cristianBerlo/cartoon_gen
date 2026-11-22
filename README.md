## ¿En qué consiste el filtro *Cartoon*?

El filtro *cartoon* transforma una fotografía o imagen estándar en una estética tipo caricatura o ilustración plana con contornos marcados mediante tres etapas principales sobre los píxeles (en formato RGB/RGBA):

1. **Suavizado preservando bordes:** Reduce el ruido y las microtexturas sin difuminar los contornos importantes de la imagen.
2. **Cuantización de color:** Reduce la cantidad de tonos disponibles por canal de color (por ejemplo, de 256 valores a 4–8 niveles discretos), generando regiones de color sólidas y uniformes.
3. **Detección y trazado de bordes:** Calcula el gradiente de intensidad y aplica un umbral para resaltar los bordes con líneas oscuras sobre la imagen cuantizada.

---

## Comparación de tecnologías y enfoques

| Enfoque | Tecnología / Estándar | Modelo de ejecución y memoria |
| :--- | :--- | :--- |
| **Secuencial** | **C** (`gcc` / `clang`) | Un solo hilo de CPU y memoria principal (`RAM`) 
| **Memoria compartida** | **C + OpenMP** | Múltiples hilos en uno o más núcleos sobre memoria compartida 
| **Memoria distribuida** | **C + MPI** | Múltiples procesos con memoria privada comunicándose por mensajes 
| **GPU** | **CUDA** (`nvcc`, C/C++) | Miles de *threads* organizados en bloques y *grids* sobre memoria de video (`VRAM`)
| **Híbrida** | **C + MPI + OpenMP** | Procesos MPI entre nodos + hilos OpenMP dentro de cada nodo 

