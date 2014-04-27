/*
============================================================================
Name : wiener1.c
Author : Tinniam V Ganesh & Egli Simon
Version :
Copyright :
Description : Wiener filter implementation in OpenCV (22 Nov 2011)
============================================================================
*/
#include <stdio.h>
#include <stdlib.h>
#include "cxcore.h"
#include "cv.h"
#include "highgui.h"

#define kappa 1
#define rad 2

int main(int argc, char ** argv)
{
int height,width,step,channels,depth;
uchar* data1;

CvMat *dft_A;
CvMat *dft_B;

CvMat *dft_C;
IplImage* im;
IplImage* im1;

IplImage* image_ReB;
IplImage* image_ImB;

IplImage* image_ReC;
IplImage* image_ImC;
IplImage* complex_ImC;
CvScalar val;

IplImage* k_image_hdr;
int i,j,k;

FILE *fp;
fp = fopen("test.txt","w+");

int dft_M,dft_N;
int dft_M1,dft_N1;

CvMat* cvShowDFT1(IplImage*, int, int,char*);
void cvShowInvDFT1(IplImage*, CvMat*, int, int,char*);

im1 = cvLoadImage( "kutty-1.jpg",1 );
cvNamedWindow("Original-color", 0);
cvShowImage("Original-color", im1);
im = cvLoadImage( "kutty-1.jpg", CV_LOAD_IMAGE_GRAYSCALE );
if( !im )
return -1;

cvNamedWindow("Original-gray", 0);
cvShowImage("Original-gray", im);

// Create a random noise matrix
fp = fopen("test.txt","w+");
int val_noise[357*383];
for(i=0; i <im->height;i++){
for(j=0;j<im->width;j++){
fprintf(fp, "%d ",(383*i+j));
val_noise[383*i+j] = rand() % 128;
}
fprintf(fp, "\n");
}

CvMat noise = cvMat(im->height,im->width, CV_8UC1,val_noise);

// Add the random noise matric to the image
cvAdd(im,&noise,im, 0);

cvNamedWindow("Original + Noise", 0);
cvShowImage("Original + Noise", im);

cvSmooth( im, im, CV_GAUSSIAN, 7, 7, 0.5, 0.5 );
cvNamedWindow("Gaussian Smooth", 0);
cvShowImage("Gaussian Smooth", im);

// Create a blur kernel
IplImage* k_image;
float r = rad;
float radius=((int)(r)*2+1)/2.0;

int rowLength=(int)(2*radius);
printf("rowlength %d\n",rowLength);
float kernels[rowLength*rowLength];
printf("rowl: %i",rowLength);
int norm=0; //Normalization factor
int x,y;
CvMat kernel;
for(x = 0; x < rowLength; x++)
for (y = 0; y < rowLength; y++)
if (sqrt((x - (int)(radius) ) * (x - (int)(radius) ) + (y - (int)(radius))* (y - (int)(radius))) <= (int)(radius))
norm++;
// Populate matrix
for (y = 0; y < rowLength; y++) //populate array with values
{
for (x = 0; x < rowLength; x++) {
if (sqrt((x - (int)(radius) ) * (x - (int)(radius) ) + (y - (int)(radius))
* (y - (int)(radius))) <= (int)(radius)) {
//kernels[y * rowLength + x] = 255;
kernels[y * rowLength + x] =1.0/norm;
printf("%f ",1.0/norm);
}
else{
kernels[y * rowLength + x] =0;
}
}
}

/*for (i=0; i < rowLength; i++){
for(j=0;j < rowLength;j++){
printf("%f ", kernels[i*rowLength +j]);
}
}*/

kernel= cvMat(rowLength, // number of rows
rowLength, // number of columns
CV_32FC1, // matrix data type
&kernels);
k_image_hdr = cvCreateImageHeader( cvSize(rowLength,rowLength), IPL_DEPTH_32F,1);
k_image = cvGetImage(&kernel,k_image_hdr);

height = k_image->height;
width = k_image->width;
step = k_image->widthStep/sizeof(float);
depth = k_image->depth;

channels = k_image->nChannels;
//data1 = (float *)(k_image->imageData);
data1 = (uchar *)(k_image->imageData);
cvNamedWindow("blur kernel", 0);
cvShowImage("blur kernel", k_image);

dft_M = cvGetOptimalDFTSize( im->height - 1 );
dft_N = cvGetOptimalDFTSize( im->width - 1 );

//dft_M1 = cvGetOptimalDFTSize( im->height+99 - 1 );
//dft_N1 = cvGetOptimalDFTSize( im->width+99 - 1 );

dft_M1 = cvGetOptimalDFTSize( im->height+3 - 1 );
dft_N1 = cvGetOptimalDFTSize( im->width+3 - 1 );

printf("dft_N1=%d,dft_M1=%d\n",dft_N1,dft_M1);

// Perform DFT of original image
dft_A = cvShowDFT1(im, dft_M1, dft_N1,"original");
//Perform inverse (check)
//cvShowInvDFT1(im,dft_A,dft_M1,dft_N1, "original"); - Commented as it overwrites the DFT

// Perform DFT of kernel
dft_B = cvShowDFT1(k_image,dft_M1,dft_N1,"kernel");
//Perform inverse of kernel (check)
//cvShowInvDFT1(k_image,dft_B,dft_M1,dft_N1, "kernel");- Commented as it overwrites the DFT

// Multiply numerator with complex conjugate
dft_C = cvCreateMat( dft_M1, dft_N1, CV_64FC2 );

printf("%d %d %d %d\n",dft_M,dft_N,dft_M1,dft_N1);

// Multiply DFT(blurred image) * complex conjugate of blur kernel
cvMulSpectrums(dft_A,dft_B,dft_C,CV_DXT_MUL_CONJ);
//cvShowInvDFT1(im,dft_C,dft_M1,dft_N1,"blur1");

// Split Fourier in real and imaginary parts
image_ReC = cvCreateImage( cvSize(dft_N1, dft_M1), IPL_DEPTH_64F, 1);
image_ImC = cvCreateImage( cvSize(dft_N1, dft_M1), IPL_DEPTH_64F, 1);
complex_ImC = cvCreateImage( cvSize(dft_N1, dft_M1), IPL_DEPTH_64F, 2);

printf("%d %d %d %d\n",dft_M,dft_N,dft_M1,dft_N1);

//cvSplit( dft_C, image_ReC, image_ImC, 0, 0 );
cvSplit( dft_C, image_ReC, image_ImC, 0, 0 );

// Compute A^2 + B^2 of denominator or blur kernel
image_ReB = cvCreateImage( cvSize(dft_N1, dft_M1), IPL_DEPTH_64F, 1);
image_ImB = cvCreateImage( cvSize(dft_N1, dft_M1), IPL_DEPTH_64F, 1);

// Split Real and imaginary parts
cvSplit( dft_B, image_ReB, image_ImB, 0, 0 );
cvPow( image_ReB, image_ReB, 2.0);
cvPow( image_ImB, image_ImB, 2.0);
cvAdd(image_ReB, image_ImB, image_ReB,0);

val = cvScalarAll(kappa);
cvAddS(image_ReB,val,image_ReB,0);

//Divide Numerator/A^2 + B^2
cvDiv(image_ReC, image_ReB, image_ReC, 1.0);
cvDiv(image_ImC, image_ReB, image_ImC, 1.0);

// Merge Real and complex parts
cvMerge(image_ReC, image_ImC, NULL, NULL, complex_ImC);

// Perform Inverse
cvShowInvDFT1(im, (CvMat *)complex_ImC,dft_M1,dft_N1,"O/p Wiener k=1 rad=2");

cvWaitKey(-1);
return 0;
}

CvMat* cvShowDFT1(IplImage* im, int dft_M, int dft_N,char* src)
{
IplImage* realInput;
IplImage* imaginaryInput;
IplImage* complexInput;

CvMat* dft_A, tmp;

IplImage* image_Re;
IplImage* image_Im;

char str[80];

double m, M;

realInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 1);
imaginaryInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 1);
complexInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 2);

cvScale(im, realInput, 1.0, 0.0);
cvZero(imaginaryInput);
cvMerge(realInput, imaginaryInput, NULL, NULL, complexInput);

dft_A = cvCreateMat( dft_M, dft_N, CV_64FC2 );
image_Re = cvCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1);
image_Im = cvCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1);

// copy A to dft_A and pad dft_A with zeros
cvGetSubRect( dft_A, &tmp, cvRect(0,0, im->width, im->height));
cvCopy( complexInput, &tmp, NULL );
if( dft_A->cols > im->width )
{
cvGetSubRect( dft_A, &tmp, cvRect(im->width,0, dft_A->cols - im->width, im->height));
cvZero( &tmp );
}

// no need to pad bottom part of dft_A with zeros because of
// use nonzero_rows parameter in cvDFT() call below

cvDFT( dft_A, dft_A, CV_DXT_FORWARD, complexInput->height );

strcpy(str,"DFT -");
strcat(str,src);
cvNamedWindow(str, 0);

// Split Fourier in real and imaginary parts
cvSplit( dft_A, image_Re, image_Im, 0, 0 );

// Compute the magnitude of the spectrum Mag = sqrt(Re^2 + Im^2)
cvPow( image_Re, image_Re, 2.0);
cvPow( image_Im, image_Im, 2.0);
cvAdd( image_Re, image_Im, image_Re, NULL);
cvPow( image_Re, image_Re, 0.5 );

// Compute log(1 + Mag)
cvAddS( image_Re, cvScalarAll(1.0), image_Re, NULL ); // 1 + Mag
cvLog( image_Re, image_Re ); // log(1 + Mag)

cvMinMaxLoc(image_Re, &m, &M, NULL, NULL, NULL);
cvScale(image_Re, image_Re, 1.0/(M-m), 1.0*(-m)/(M-m));
cvShowImage(str, image_Re);
return(dft_A);
}

void cvShowInvDFT1(IplImage* im, CvMat* dft_A, int dft_M, int dft_N,char* src)
{

IplImage* realInput;
IplImage* imaginaryInput;
IplImage* complexInput;

IplImage * image_Re;
IplImage * image_Im;

double m, M;
char str[80];

realInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 1);
imaginaryInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 1);
complexInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 2);

image_Re = cvCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1);
image_Im = cvCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1);

//cvDFT( dft_A, dft_A, CV_DXT_INV_SCALE, complexInput->height );
cvDFT( dft_A, dft_A, CV_DXT_INV_SCALE, dft_M);
strcpy(str,"DFT INVERSE - ");
strcat(str,src);
cvNamedWindow(str, 0);

// Split Fourier in real and imaginary parts
cvSplit( dft_A, image_Re, image_Im, 0, 0 );

// Compute the magnitude of the spectrum Mag = sqrt(Re^2 + Im^2)
cvPow( image_Re, image_Re, 2.0);
cvPow( image_Im, image_Im, 2.0);
cvAdd( image_Re, image_Im, image_Re, NULL);
cvPow( image_Re, image_Re, 0.5 );

// Compute log(1 + Mag)
cvAddS( image_Re, cvScalarAll(1.0), image_Re, NULL ); // 1 + Mag
cvLog( image_Re, image_Re ); // log(1 + Mag)

cvMinMaxLoc(image_Re, &m, &M, NULL, NULL, NULL);
cvScale(image_Re, image_Re, 1.0/(M-m), 1.0*(-m)/(M-m));
//cvCvtColor(image_Re, image_Re, CV_GRAY2RGBA);

cvShowImage(str, image_Re);

}


