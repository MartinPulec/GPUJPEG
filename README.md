GPUJPEG
=======
[![@gpujpeg:matrix.org](https://img.shields.io/badge/Matrix-chat-black)](https://matrix.to/#/!ppSneXxiHfvznPxTTN:matrix.org?via=matrix.org)

**JPEG** encoder and decoder library and console application for **NVIDIA GPUs**.

This documents provides an introduction to the library and how to use it. You
can also look at [FAQ](FAQ.md) for additional information.

Table of contents
-----------------
- [Authors](#authors)
- [Description](#description)
- [Overview](#overview)
- [Performance](#performance)
  * [Encoding 4K (4096x2160) and HD (1920x1080)](#encoding-4k-4096x2160-and-hd-1920x1080)
  * [Decoding 4K (4096x2160) and HD (1920x1080)](#decoding-4k-4096x2160-and-hd-1920x1080)
- [Compile](#compile)
- [Usage](#usage)
  * [libgpujpeg library](#libgpujpeg-library)
    + [Encoding](#encoding)
    + [Decoding](#decoding)
  * [GPUJPEG console application](#gpujpeg-console-application)
- [Requirements](#requirements)
- [License](#license)
- [References](#references)

Authors
-------
 - Martin Srom, CESNET z.s.p.o
 - Jan Brothánek
 - Martin Jirman
 - Jiri Matela
 - Martin Pulec
 - Petr Holub

Description
-----------
The first implementation of the JPEG image compression standard for
NVIDIA GPUs used for real-time transmission of high-definition video.

Overview
--------
- It uses NVIDIA CUDA platform.
- baseline Huffman 8-bit coding, JFIF file format (or Adobe for RGB)
- use of _restart markers_ that allow fast parallel encoding/decoding
- Encoder by default creates non-interleaved stream, optinally it can produce
  an interleaved stream (all components in one scan) or/and subsampled stream.
- support for color transformations and coding RGB JPEG
- Decoder can decompress JPEG codestreams that can be generated by encoder. If scan
  contains restart flags, decoder can use parallelism for fast decoding.
- Encoding/Decoding of JPEG codestream is divided into following phases:

       Encoding:                       Decoding
       1) Input data loading           1) Input data loading
       2) Preprocessing                2) Parsing codestream
       3) Forward DCT                  3) Huffman decoder
       4) Huffman encoder              4) Inverse DCT
       5) Formatting codestream        5) Postprocessing

  and they are implemented on CPU or/and GPU as follows:
   - CPU:
      - Input data loading
      - Parsing codestream
      - Huffman encoder/decoder (when restart flags are disabled)
      - Output data formatting
   - GPU:
      - Preprocessing/Postprocessing (color component parsing,
        color transformation RGB <-> YCbCr)
      - Forward/Inverse DCT (discrete cosine transform)
      - Huffman encoder/decoder (when restart flags are enabled)

Performance
-----------
Following tables summarizes encoding/decoding performance using NVIDIA
GTX 580 for non-interleaved and non-subsampled stream with different quality
settings (time, PSNR and encoded size values are averages of encoding several
images, each of them multiple times):

### Encoding 4K (4096x2160) and HD (1920x1080)
 quality | duration 4K |  PSNR 4K |   size 4K  | duration HD |  PSNR HD |  size HD
 --------|-------------|----------|------------|-------------|----------|-----------
   10    |   26.79 ms  | 29.33 dB |  539.30 kB |    6.71 ms  | 27.41 dB |  145.90 kB
   20    |   26.91 ms  | 32.70 dB |  697.20 kB |    6.74 ms  | 30.32 dB |  198.30 kB
   30    |   27.17 ms  | 34.63 dB |  850.60 kB |    6.84 ms  | 31.92 dB |  243.60 kB
   40    |   27.19 ms  | 35.97 dB |  958.90 kB |    6.89 ms  | 32.99 dB |  282.20 kB
   50    |   27.29 ms  | 36.94 dB | 1073.30 kB |    6.92 ms  | 33.82 dB |  319.10 kB
   60    |   27.39 ms  | 37.96 dB | 1217.10 kB |    6.95 ms  | 34.65 dB |  360.00 kB
   70    |   27.51 ms  | 39.22 dB | 1399.20 kB |    7.04 ms  | 35.71 dB |  422.10 kB
   80    |   27.76 ms  | 40.67 dB | 1710.00 kB |    7.13 ms  | 37.15 dB |  526.70 kB
   90    |   28.36 ms  | 42.83 dB | 2441.40 kB |    7.32 ms  | 39.84 dB |  768.40 kB
  100    |   35.47 ms  | 47.09 dB | 7798.70 kB |    9.31 ms  | 47.21 dB | 2499.60 kB

### Decoding 4K (4096x2160) and HD (1920x1080)
 quality | duration 4K |  PSNR 4K |   size 4K  | duration 4K |  PSNR 4K |  size 4K
 --------|-------------|----------|------------|-------------|----------|-----------
   10    |   10.28 ms  | 29.33 dB |  539.30 kB |    3.13 ms  | 27.41 dB |  145.90 kB
   20    |   11.31 ms  | 32.70 dB |  697.20 kB |    3.59 ms  | 30.32 dB |  198.30 kB
   30    |   12.36 ms  | 34.63 dB |  850.60 kB |    3.97 ms  | 31.92 dB |  243.60 kB
   40    |   12.90 ms  | 35.97 dB |  958.90 kB |    4.28 ms  | 32.99 dB |  282.20 kB
   50    |   13.45 ms  | 36.94 dB | 1073.30 kB |    4.56 ms  | 33.82 dB |  319.10 kB
   60    |   14.71 ms  | 37.96 dB | 1217.10 kB |    4.81 ms  | 34.65 dB |  360.00 kB
   70    |   15.03 ms  | 39.22 dB | 1399.20 kB |    5.24 ms  | 35.71 dB |  422.10 kB
   80    |   16.64 ms  | 40.67 dB | 1710.00 kB |    5.89 ms  | 37.15 dB |  526.70 kB
   90    |   19.99 ms  | 42.83 dB | 2441.40 kB |    7.48 ms  | 39.84 dB |  768.40 kB
  100    |   46.45 ms  | 47.09 dB | 7798.70 kB |   16.42 ms  | 47.21 dB | 2499.60 kB

Compile
-------
To build console application check [Requirements](#requirements) and go
to `gpujpeg` directory (where [README.md](README.md) and [COPYING](COPYING)
files are placed) and run `autogen.sh` and `make` command. It builds
libgpugjpeg library in subdirectory `.libs` and it creates script `gpujpeg`
which runs executable file linked to runtime library _libgpujpeg.so_
(which is placed in `.libs` subdirectory).

You can also use **CMake** to crate build recipe for the library and the
application (including project file for MSVS) or a plain old _Makefile.bkp_.

In _Windows_ you may want to avoid using build system at all and use the
following command directly:

    nvcc -I. -DGPUJPEG_EXPORTS -o gpujpeg.dll --shared src/gpujpeg_*c src/gpujpeg*cu # library
    nvcc -I. -o gpujpeg src/*c src/*cu # console application

or (in _Linux_):

    nvcc -I. -Xcompiler -fPIC -o gpujpeg.so --shared src/gpujpeg_*c src/gpujpeg*cu # library
    nvcc -I -o gpujpeg src/*c src/*cu # console application

Usage
-----

### libgpujpeg library
To build _libgpujpeg_ library check [Compile](#compile).

To use library in your project you have to include library to your
sources and linked shared library object to your executable:

    #include "libgpujpeg/gpujpeg.h"

#### Encoding
For encoding by libgpujpeg library you have to declare two structures
and set proper values to them. The first is definition of encoding/decoding
parameters, and the second is structure with parameters of input image:

    struct gpujpeg_parameters param;
    gpujpeg_set_default_parameters(&param);
    param.quality = 80;
    // (default value is 75)
    param.restart_interval = 16;
    // (default value is 8)
    param.interleaved = 1;
    // (default value is 0)
    
    struct gpujpeg_image_parameters param_image;
    gpujpeg_image_set_default_parameters(&param_image);
    param_image.width = 1920;
    param_image.height = 1080;
    param_image.comp_count = 3;
    // (for now, it must be 3)
    param_image.color_space = GPUJPEG_RGB;
    // or GPUJPEG_BT709 or GPUJPEG_YCBCR_JPEG
    // (default value is GPUJPEG_RGB)
    param_image.pixel_format = GPUJPEG_444_U8_P012;
    // or eg. GPUJPEG_422_U8_P1020
    // (default value is GPUJPEG_444_U8_P012)

If you want to use subsampling in JPEG format call following function,
that will set default sampling factors (2x2 for Y, 1x1 for Cb and Cr):

    // Use default sampling factors
    gpujpeg_parameters_chroma_subsampling(&param);

Or define sampling factors by hand:

    // User custom sampling factors
    param.sampling_factor[0].horizontal = 4;
    param.sampling_factor[0].vertical = 4;
    param.sampling_factor[1].horizontal = 1;
    param.sampling_factor[1].vertical = 2;
    param.sampling_factor[2].horizontal = 2;
    param.sampling_factor[2].vertical = 1;

Next you have to initialize CUDA device by calling:

    if ( gpujpeg_init_device(device_id, 0) )
        return -1;

where first parameters is CUDA device (e.g. `device_id = 0`) id and second
parameter is flag if verbose output should be used (`0` or `GPUJPEG_VERBOSE`).
Next step is to create encoder:

    struct gpujpeg_encoder* encoder = gpujpeg_encoder_create(&param,
        &param_image);
    if ( encoder == NULL )
        return -1;

When creating encoder, library allocates all device buffers which will be
needed for image encoding and when you encode concrete image, they are
already allocated and encoder will used them for every image. Now we need
raw image data that we can encode by encoder, for example we can load it
from file:

    int image_size = 0;
    uint8_t* input_image = NULL;
    if ( gpujpeg_image_load_from_file("input_image.rgb", &input_image,
             &image_size) != 0 )
        return -1;

Next step is to encode uncompressed image data to JPEG compressed data
by encoder:

    struct gpujpeg_encoder_input encoder_input;
    gpujpeg_encoder_input_set_image(&encoder_input, input_image);

    uint8_t* image_compressed = NULL;
    int image_compressed_size = 0;
    if ( gpujpeg_encoder_encode(encoder, &encoder_input, &image_compressed,
             &image_compressed_size) != 0 )
        return -1;

Compressed data are placed in internal encoder buffer so we have to save
them somewhere else before we start encoding next image, for example we
can save them to file:

    if ( gpujpeg_image_save_to_file("output_image.jpg", image_compressed,
             image_compressed_size) != 0 )
        return -1;

Now we can load, encode and save next image or finish and move to clean up
encoder. Finally we have to clean up so destroy loaded image and destroy
the encoder.

    gpujpeg_image_destroy(input_image);
    gpujpeg_encoder_destroy(encoder);

#### Decoding
For decoding we don't need to initialize two structures of parameters.
We only have to initialize CUDA device if we haven't initialized it yet and
create decoder:

    if ( gpujpeg_init_device(device_id, 0) )
        return -1;
    
    struct gpujpeg_decoder* decoder = gpujpeg_decoder_create(NULL);
    if ( decoder == NULL )
        return -1;

Now we have two options. The first is to do nothing and decoder will
postpone buffer allocations to decoding first image where it determines
proper image size and all other parameters. All the following images must
have the same parameters. The second option is to provide input image size
and optionally other parameters and the decoder will allocate all buffers
and it is fully ready when encoding even the first image.

    struct gpujpeg_parameters param;
    gpujpeg_set_default_parameters(&param);
    param.restart_interval = 16;
    param.interleaved = 1;
    
    struct gpujpeg_image_parameters param_image;
    gpujpeg_image_set_default_parameters(&param_image);
    param_image.width = 1920;
    param_image.height = 1080;
    param_image.comp_count = 3;
    
    // Pre initialize decoder before decoding
    gpujpeg_decoder_init(decoder, &param, &param_image);

If you want to specify output image color space and/or subsampling factor,
you can use following two parameters. You can specify them though the
param structure befor passing it to `gpujpeg_decoder_init`. But if you
postpone this initialization process to the first image, you have no
other option than specify them in this way:

    gpujpeg_decoder_set_output_format(decoder, GPUJPEG_RGB,
                    GPUJPEG_444_U8_P012);
    // or eg. GPUJPEG_YCBCR_JPEG and GPUJPEG_422_U8_P1020

Next we have to load JPEG image data from file and decoded it to raw
image data:

    int image_size = 0;
    uint8_t* image = NULL;
    if ( gpujpeg_image_load_from_file("input_image.jpg", &image,
             &image_size) != 0 )
        return -1;
    
    struct gpujpeg_decoder_output decoder_output;
    gpujpeg_decoder_output_set_default(&decoder_output);
    if ( gpujpeg_decoder_decode(decoder, image, image_size,
             &decoder_output) != 0 )
        return -1;

Now we can save decoded raw image data to file and perform cleanup:

    if ( gpujpeg_image_save_to_file("output_image.rgb", decoder_output.data,
             decoder_output.data_size) != 0 )
        return -1;
    
    gpujpeg_image_destroy(image);
    gpujpeg_decoder_destroy(decoder);

### GPUJPEG console application
The console application gpujpeg uses _libgpujpeg_ library to demonstrate
it's functions. To build console application check [Compile](#compile).

To encode image from raw RGB image file to JPEG image file use following
command:

    ./gpujpeg --encode --size=WIDTHxHEIGHT --quality=QUALITY \
            INPUT_IMAGE.rgb OUTPUT_IMAGE.jpg

You must specify input image size by `--size=WIDTHxHEIGHT` parameter.
Optionally you can specify desired output quality by parameter
`--quality=QUALITY` which accepts values 0-100. Console application accepts
a few more parameters and you can list them by folling command:

    ./gpujpeg --help

To decode image from JPEG image file to raw RGB image file use following
command:

    ./gpujpeg --decode OUTPUT_IMAGE.jpg INPUT_IMAGE.rgb

You can also encode and decode image to test the console application:

    ./gpujpeg --encode --decode --size=WIDTHxHEIGHT --quality=QUALITY \
            INPUT_IMAGE.rgb OUTPUT_IMAGE.jpg

Decoder will create new decoded file `OUTPUT_IMAGE.jpg.decoded.rgb` and do
not overwrite your `INPUT_IMAGE.rgb` file.

Console application is able to load raw RGB image file data from *.rgb
files and raw YUV and YUV422 data from *.yuv files. For YUV422 you must
specify *.yuv file and use `--sampling-factor=4:2:2` parameter.

All supported parameters for console application are following:

    --help
        Prints console application help
    --size=1920x1080
        Input image size in pixels, e.g. 1920x1080
    --pixel-format=444-u8-p012
        Input/output image pixel format ('u8', '444-u8-p012', '444-u8-p012z',
        '444-u8-p0p1p2', '422-u8-p1020', '422-u8-p0p1p2' or '420-u8-p0p1p2')
    --colorspace=rgb
        Input image colorspace (supported are 'rgb', 'yuv' and 'ycbcr-jpeg',
        where 'yuv' means YCbCr ITU-R BT.601), when *.yuv file is specified,
        instead of default 'rgb', automatically the colorspace 'yuv' is used
    --quality
        Set output quality level 0-100 (default 75)
    --restart=8
        Set restart interval for encoder, number of MCUs between
        restart markers
    --subsampled
        Produce chroma subsampled JPEG stream
    --interleaved
        Produce interleaved stream
    --encode
        Encode images
    --decode
        Decode images
    --device=0
        By using this parameter you can specify CUDA device id which will
        be used for encoding/decoding.

Restart interval is important for parallel huffman encoding and decoding.
When `--restart=N` is used (default is 8), the coder can process each
N MCUs independently, and so he can code each N MCUs in parallel. When
`--restart=0` is specified, restart interval is disabled and the coder
must use CPU version of huffman coder (because on GPU would run only one
thread, which is very slow).

The console application can encode/decode multiple images by following
command:

    ./gpujpeg ARGUMENTS INPUT_IMAGE_1.rgb OUTPUT_IMAGE_1.jpg \
            INPUT_IMAGE_2.rgb OUTPUT_IMAGE_2.jpg ...

Requirements
------------
To be able to build and run libgpujpeg **library** and gpujpeg **console
application** you need:

1) CUDA Toolkit (http://developer.nvidia.com/cuda-toolkit) installed,
   default installation path is `/usr/local/cuda`. If you have the CUDA
   installed somewhere else, you need to specify it by environment variable
   `CUDA_INSTALL_PATH` or in Makefiles by `CUDA_INSTALL_PATH` variable.
2) NVIDIA developer drivers
3) CUDA enabled NVIDIA GPU

License
-------
- See file [COPYING](COPYING).
- This software contains source code provided by NVIDIA Corporation.
- This software source code is based on SiGenGPU \[[3]\].

References
----------
[1]: http://www.w3.org/Graphics/JPEG/itu-t81.pdf
[2]: http://www.ijg.org/
[3]: https://github.com/silicongenome/SiGenGPU
[4]: https://www.ecma-international.org/publications/files/ECMA-TR/ECMA%20TR-098.pdf

1. [ITU T-81][1]
2. [ILG][2]
3. [SiGenGPU][3] (currently defunct)
4. [ECMA TR/098 (JFIF)][4]

