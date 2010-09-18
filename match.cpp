#include "sift.h"
#include "imgfeatures.h"
#include "kdtree.h"
#include "utils.h"
#include "xform.h"
#include "gpc.h"
#include "SiftTree.h"

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_permutation.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <vector>

#ifdef _DEBUG
#define COMPLAIN_OUT_OF_BOUND 0
#endif

/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

/* threshold on squared ratio of distances between NN and 2nd NN */
#define NN_SQ_DIST_RATIO_THR 0.49

#define EQUAL_IMAGE_SIZE(src, dst) (((src)->width == (dst)->width)&&((dst)->height == (dst)->height))

#define INTER_TIMES_MAX 50
#define IMPROVE_VALUE_THRESHOLD 0.05
#define IMPROVE_RATIO_THRESHOLD 0.005

char *img1_file = 0;
char *img2_file = 0;

double imgzoom_scale=1.0;

IplImage* stacked;

/****** callbacks *******/
void on_mouse( int event, int x, int y, int flags, void* param ) ;
void resize_img();

/**
 * Combine two images to construct one image.
 */
void OverlayImages(const IplImage *part1, double w1, const IplImage *part2, double w2, IplImage *out);

/**
 * Blend images using algorithm proposed by Burt and Adelson. 
 * Please refer to "A Multiresolution Spline With Application to Image Mosaics" for more info.
 */
template <typename ElementType>
void blend_images(const IplImage *part1, const IplImage *part2, const IplImage *map, IplImage *out);

/**
 * Warp by a transformation matrix. src(x', y') = dst(x, y), and t*[x', y', 1]' = transMat * [x, y, 1]'.
 * Because (x', y') may be negative, we add a pre-calculated shift to make sure all image coordinates be
 * positive.
 */
template <typename ElementType>
void warp_by_matrix(IplImage *src, IplImage *dst, CvMat *transMat, CvPoint shift);

/**
 * Shift source image to fit into the dst image.
 */
template <typename ElementType>
void shift_image(const IplImage *src, IplImage *dst, CvPoint shift);

/**
 * Given an initial transform matrix estimate and feature point pairs, 
 * then use leverberg-marquet algorithm to optimize the result.
 */
void nonlinear_optimize(CvMat *trans, struct feature **in, int nin);

CvPoint ProjPoint(CvPoint in, CvMat *trans);

int MatchSiftFeats(struct kd_node **ppKdRoot, 
        struct feature *pFeat1, int n1, 
        struct feature *pFeat2, int n2,
        struct CvSeq *pUpSeq, 
        struct CvSeq *pDownSeq,
        int    img1Height)
{
    struct feature   *feat;
    struct feature** nbrs;
    CvPoint pt1, pt2;

    int i, k, m=0;
    double d0, d1;

    *ppKdRoot = kdtree_build( pFeat2, n2 );

    for( i = 0; i < n1; i++ )
    {
        feat = pFeat1 + i;
        k = kdtree_bbf_knn( *ppKdRoot, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );
        if( k == 2 )
        {
            d0 = descr_dist_sq( feat, nbrs[0] );
            d1 = descr_dist_sq( feat, nbrs[1] );
            if( d0 < d1 * NN_SQ_DIST_RATIO_THR )
            {
                pt1 = cvPoint( cvRound( feat->x ), cvRound( feat->y ) );
                pt2 = cvPoint( cvRound( nbrs[0]->x ), cvRound( nbrs[0]->y ) );

                cvSeqPush(pUpSeq,  &pt1); 
                cvSeqPush(pDownSeq,&pt2);

                pt2.y += img1Height;
                cvLine( stacked, pt1, pt2, CV_RGB(255,0,255), 1, 8, 0 );
                m++;
                pFeat1[i].fwd_match = nbrs[0];
            }
        }

        free( nbrs );
    }

    return 0;
}

int InverseTransMat(CvMat **ppInvMat, const CvMat *pMat)
{
    double ret;

    *ppInvMat = cvCreateMat(pMat->rows, pMat->cols, pMat->type);
    ret = cvInvert(pMat, *ppInvMat, CV_LU);

    if (ret == 0)
        return -1;
    else
        return 0;
}

int IntersectTwoImages(gpc_polygon *intersect, 
        const IplImage *img1, 
        const IplImage *img2, 
        const CvMat *transMat)
{
    int ret;

    /* Get transform matrix from img2 to img1 */
    CvMat *invTransMat = NULL;
    ret = InverseTransMat(&invTransMat, transMat);
    assert(ret == 0);

    CvMat *tmpVector   = cvCreateMat(3, 1, CV_64FC1);
    CvMat *resltVector = cvCreateMat(3, 1, CV_64FC1);

    gpc_polygon invImg2Bound;
    gpc_polygon img1Bound;

    CvPoint2D64f img1VertLst[4];
    CvPoint2D64f img2VertLst[4];

    img1VertLst[0].x = 0;
    img1VertLst[0].y = 0;
    img1VertLst[1].x = img1->width;
    img1VertLst[1].y = 0;
    img1VertLst[2].x = img1->width;
    img1VertLst[2].y = img1->height;
    img1VertLst[3].x = 0;
    img1VertLst[3].y = img1->height;

    img2VertLst[0].x = 0;
    img2VertLst[0].y = 0;
    img2VertLst[1].x = img2->width;
    img2VertLst[1].y = 0;
    img2VertLst[2].x = img2->width;
    img2VertLst[2].y = img2->height;
    img2VertLst[3].x = 0;
    img2VertLst[3].y = img2->height;

    // Convert img2 corners use "invTransMat x [corner.x, corner.y, 1]^T"
    int i;
    for (i=0; i<4; i++)
    {
        cvSetReal1D(tmpVector, 0, img2VertLst[i].x);
        cvSetReal1D(tmpVector, 1, img2VertLst[i].y);
        cvSetReal1D(tmpVector, 2, 1.0);

        cvMatMul(invTransMat, tmpVector, resltVector);

        double scale = cvGetReal1D(resltVector, 2);
        cvScale(resltVector, resltVector, 1.0/scale);

        img2VertLst[i].x = cvGetReal1D(resltVector, 0);
        img2VertLst[i].y = cvGetReal1D(resltVector, 1);
    }

    //int poly1Fd, poly2Fd;
    //char tmpFlTmplt1[] = "poly_XXXXXX";
    //char tmpFlTmplt2[] = "poly_XXXXXX";
    FILE *poly1Fp, *poly2Fp;

    //poly1Fd = mkstemp(tmpFlTmplt1);
    //poly2Fd = mkstemp(tmpFlTmplt2);
    //poly1Fp = fdopen(poly1Fd, "w+");
    //poly2Fp = fdopen(poly2Fd, "w+");
    
    poly1Fp = fopen("poly1_vertices", "w+");
    poly2Fp = fopen("poly2_vertices", "w+");

    fprintf(poly1Fp, "%d\n", 1);
    fprintf(poly1Fp, "%d\n", 4);
    fprintf(poly2Fp, "%d\n", 1);
    fprintf(poly2Fp, "%d\n", 4);
    for (i=0; i<4; i++)
    {
        fprintf(poly1Fp, "%lf %lf\n", img1VertLst[i].x, img1VertLst[i].y);
        fprintf(poly2Fp, "%lf %lf\n", img2VertLst[i].x, img2VertLst[i].y);
    }

    fseek(poly1Fp, 0, SEEK_SET);
    fseek(poly2Fp, 0, SEEK_SET);

    gpc_read_polygon(poly1Fp, 0, &img1Bound);
    gpc_read_polygon(poly2Fp, 0, &invImg2Bound);

    gpc_polygon_clip(GPC_INT, &img1Bound, &invImg2Bound, intersect);

    FILE *intFp = fopen("poly_inter", "w");
    gpc_write_polygon(intFp, 0, intersect);

    cvReleaseMat(&invTransMat);
    cvReleaseMat(&tmpVector);
    cvReleaseMat(&resltVector);
    gpc_free_polygon(&img1Bound);
    gpc_free_polygon(&invImg2Bound);
    
    fclose(poly1Fp);
    fclose(poly2Fp);

    return 0;
}


int FindContainRect(CvRect *rect, const gpc_polygon *intersect, const CvSize *img1Sz)
{
    double upLeft[2], downRight[2];
    upLeft[0] = img1Sz->width;
    upLeft[1] = img1Sz->height;
    downRight[0] = 0;
    downRight[1] = 0;

    assert(intersect->num_contours > 0);

    int i;
    int numVertices = intersect->contour->num_vertices;
    gpc_vertex *pVertices = intersect->contour->vertex;
    for (i=0; i<numVertices; i++)
    {
        if (pVertices[i].x < upLeft[0])
            upLeft[0] = pVertices[i].x;
        if (pVertices[i].y < upLeft[1])
            upLeft[1] = pVertices[i].y;

        if (pVertices[i].x > downRight[0])
            downRight[0] = pVertices[i].x;
        if (pVertices[i].y > downRight[1])
            downRight[1] = pVertices[i].y;
    }

    // Must be in img1
    if (upLeft[0] < 0.0)
        upLeft[0] = 0;
    if (upLeft[1] < 0.0)
        upLeft[1] = 0;
    if (downRight[0] > img1Sz->width)
        downRight[0] = img1Sz->width;
    if (downRight[1] > img1Sz->height)
        downRight[1] = img1Sz->height;

    rect->x = (int)upLeft[0];
    rect->y = (int)upLeft[1];
    rect->width  = ceil(downRight[0]) - rect->x;
    rect->height = ceil(downRight[1]) - rect->y;

    return 0;
}


inline double InterpImgGrayScale(const CvMat *pImgMat, double dr, double dc)
{
    double coffRight, coffLeft;
    double coffUp, coffDown;

    int tmpx, tmpy;
    tmpx      = floor(dc);
    coffLeft  = dc - tmpx;
    coffRight = 1.0 - coffLeft;

    tmpy      = floor(dr);
    coffUp    = dr - tmpy;
    coffDown  = 1.0 - coffUp;

    double rslt = 0.0;
    rslt += cvGetReal2D(pImgMat, tmpy, tmpx)*coffRight*coffDown;
    rslt += cvGetReal2D(pImgMat, tmpy+1, tmpx)*coffRight*coffUp;
    rslt += cvGetReal2D(pImgMat, tmpy, tmpx+1)*coffLeft*coffDown;
    rslt += cvGetReal2D(pImgMat, tmpy+1, tmpx+1)*coffLeft*coffUp;

    return rslt;
}


int MapRoi(CvMat *rslt, 
           const CvRect *roi, const CvMat *transMat, const CvSize *img2Sz)
{
    int i, j;
    int num;
    double *dbMat;

    int depth, cn;
    depth = CV_MAT_DEPTH(transMat->type);
    cn    = CV_MAT_CN(transMat->type);
    assert(depth == CV_64F);
    assert(cn == 1);

    num = 0;
    dbMat = transMat->data.db;
    for (i=0; i<roi->height; i++)
    {
        for (j=0; j<roi->width; j++)
        {
            double c1, c2, c3;
            c1 = dbMat[0]*(j+roi->x) + dbMat[1]*(i+roi->y) + dbMat[2];
            c2 = dbMat[3]*(j+roi->x) + dbMat[4]*(i+roi->y) + dbMat[5];
            c3 = dbMat[6]*(j+roi->x) + dbMat[7]*(i+roi->y) + dbMat[8];

            CvScalar elmt;
            elmt.val[0] = c1;
            elmt.val[1] = c2;
            elmt.val[2] = c3;
            cvSet2D(rslt, i, j, elmt);

            c1 /= c3;
            c2 /= c3;
            c3 = 1.0;

            if (c1 < 0 || c1 > (img2Sz->width -1) ||
                c2 < 0 || c2 > (img2Sz->height-1))
            {
                double *elmt = (double *)cvPtr2D(rslt, i, j, NULL);
                elmt[2] = 0.0;
            }
            else
            {
                num += 1;
            }
        }
    }

    return num;
}


int CalcNewtonVector(CvMat *pNewtonVect, double *pErr,
        const IplImage *img1,   const IplImage *img2, 
        const IplImage *img2dx, const IplImage *img2dy, 
        const IplImage *img2dxdx,
        const IplImage *img2dydy,
        const IplImage *img2dxdy,
        const CvRect *pRoi,     
        const CvMat  *pProjRslt,
        int   validNum)
{
    int i, j;
    *pErr = 0.0;

    CvMat img1Mat, img2Mat;
    cvGetMat(img1, &img1Mat, NULL, NULL);
    cvGetMat(img2, &img2Mat, NULL, NULL);
    
    CvMat img2dxMat, img2dyMat;
    cvGetMat(img2dx, &img2dxMat, NULL, NULL);
    cvGetMat(img2dy, &img2dyMat, NULL, NULL);

    CvMat img2dxdxMat, img2dydyMat, img2dxdyMat;
    cvGetMat(img2dxdx, &img2dxdxMat, NULL, NULL);
    cvGetMat(img2dxdy, &img2dxdyMat, NULL, NULL);
    cvGetMat(img2dydy, &img2dydyMat, NULL, NULL);

    CvMat *pDx2Da =   cvCreateMat(8, 1, CV_64FC1);
    CvMat *pDy2Da =   cvCreateMat(8, 1, CV_64FC1);
    CvMat *pDx2DaDa = cvCreateMat(8, 8, CV_64FC1);
    CvMat *pDy2DaDa = cvCreateMat(8, 8, CV_64FC1);

    CvMat *pGradnt  = cvCreateMat(8, 1, CV_64FC1);
    CvMat *pSecDerv = cvCreateMat(8, 8, CV_64FC1);
    CvMat *pInterim = cvCreateMat(8, 1, CV_64FC1);

    cvSetZero(pNewtonVect);
    cvSetZero(pGradnt);
    cvSetZero(pSecDerv);

    for (i=pRoi->y; i<pRoi->y+pRoi->height; i++)
    {
        for (j=pRoi->x; j<pRoi->x+pRoi->width; j++)
        {
            int r, c;
            r = i - pRoi->y;
            c = j - pRoi->x;

            double dx, dy, dz;
            CvScalar projElmt = cvGet2D(pProjRslt, r, c);
            dx = projElmt.val[0];
            dy = projElmt.val[1];
            dz = projElmt.val[2];
            if (0.0 == dz)
                continue;

            double nx, ny;
            nx = dx / dz;
            ny = dy / dz;

            cvZero(pDx2Da);
            cvZero(pDy2Da);

            double grayScaleDiff;
            grayScaleDiff = cvGetReal2D(&img1Mat, i, j) 
                - InterpImgGrayScale(&img2Mat, ny, nx);

            double img2Dx, img2Dy;
            img2Dx = InterpImgGrayScale(&img2dxMat, ny, nx); 
            img2Dy = InterpImgGrayScale(&img2dyMat, ny, nx);

            double img2DxDx, img2DyDy, img2DxDy;
            img2DxDx = InterpImgGrayScale(&img2dxdxMat, ny, nx);
            img2DyDy = InterpImgGrayScale(&img2dydyMat, ny, nx);
            img2DxDy = InterpImgGrayScale(&img2dxdyMat, ny, nx);

            // First calculate dx_2/dA
            cvSetReal1D(pDx2Da, 0, j/dz);
            cvSetReal1D(pDx2Da, 1, i/dz);
            cvSetReal1D(pDx2Da, 2, 1.0/dz);
            cvSetReal1D(pDx2Da, 3, 0.0);
            cvSetReal1D(pDx2Da, 4, 0.0);
            cvSetReal1D(pDx2Da, 5, 0.0);
            cvSetReal1D(pDx2Da, 6, -dx*j/dz/dz);
            cvSetReal1D(pDx2Da, 7, -dx*i/dz/dz);
            
            // Second calculate dy_2/dA
            cvSetReal1D(pDy2Da, 0, 0.0);
            cvSetReal1D(pDy2Da, 1, 0.0);
            cvSetReal1D(pDy2Da, 2, 0.0);
            cvSetReal1D(pDy2Da, 3, j/dz);
            cvSetReal1D(pDy2Da, 4, i/dz);
            cvSetReal1D(pDy2Da, 5, 1.0/dz);
            cvSetReal1D(pDy2Da, 6, -dy*j/dz/dz);
            cvSetReal1D(pDy2Da, 7, -dy*i/dz/dz);

            // Third calculate d^2x_2/dAdA
            cvZero(pDx2DaDa);
            cvSetReal2D(pDx2DaDa, 0, 6, -j*j/dz/dz);
            cvSetReal2D(pDx2DaDa, 1, 6, -j*i/dz/dz);
            cvSetReal2D(pDx2DaDa, 2, 6, -j/dz/dz);
            cvSetReal2D(pDx2DaDa, 0, 7, -j*i/dz/dz);
            cvSetReal2D(pDx2DaDa, 1, 7, -i*i/dz/dz);
            cvSetReal2D(pDx2DaDa, 2, 7, -i/dz/dz);
            cvSetReal2D(pDx2DaDa, 6, 0, -j*j/dz/dz);
            cvSetReal2D(pDx2DaDa, 6, 1, -j*i/dz/dz);
            cvSetReal2D(pDx2DaDa, 6, 2, -j/dz/dz);
            cvSetReal2D(pDx2DaDa, 7, 0, -j*i/dz/dz);
            cvSetReal2D(pDx2DaDa, 7, 1, -i*i/dz/dz);
            cvSetReal2D(pDx2DaDa, 7, 2, -i/dz/dz);
            cvSetReal2D(pDx2DaDa, 6, 6, 2*dx*j*j/dz/dz/dz);
            cvSetReal2D(pDx2DaDa, 6, 7, 2*dx*j*i/dz/dz/dz);
            cvSetReal2D(pDx2DaDa, 7, 6, 2*dx*j*i/dz/dz/dz);
            cvSetReal2D(pDx2DaDa, 7, 7, 2*dx*i*i/dz/dz/dz);

            // Third calculate d^2x_2/dAdA
            cvZero(pDy2DaDa);
            cvSetReal2D(pDy2DaDa, 3, 6, -j*j/dz/dz);
            cvSetReal2D(pDy2DaDa, 4, 6, -j*i/dz/dz);
            cvSetReal2D(pDy2DaDa, 5, 6, -j/dz/dz);
            cvSetReal2D(pDy2DaDa, 3, 7, -j*i/dz/dz);
            cvSetReal2D(pDy2DaDa, 4, 7, -i*i/dz/dz);
            cvSetReal2D(pDy2DaDa, 5, 7, -i/dz/dz);
            cvSetReal2D(pDy2DaDa, 6, 3, -j*j/dz/dz);
            cvSetReal2D(pDy2DaDa, 6, 4, -j*i/dz/dz);
            cvSetReal2D(pDy2DaDa, 6, 5, -j/dz/dz);
            cvSetReal2D(pDy2DaDa, 7, 3, -j*i/dz/dz);
            cvSetReal2D(pDy2DaDa, 7, 4, -i*i/dz/dz);
            cvSetReal2D(pDy2DaDa, 7, 5, -i/dz/dz);
            cvSetReal2D(pDy2DaDa, 6, 6, 2*dy*j*j/dz/dz/dz);
            cvSetReal2D(pDy2DaDa, 6, 7, 2*dy*j*i/dz/dz/dz);
            cvSetReal2D(pDy2DaDa, 7, 6, 2*dy*j*i/dz/dz/dz);
            cvSetReal2D(pDy2DaDa, 7, 7, 2*dy*i*i/dz/dz/dz);
            
            // Need -2.0*grayScaleDiff/validNum later
            cvAddWeighted(pDx2Da, img2Dx, 
                    pDy2Da, img2Dy,
                    0, pInterim);
            // Gradient
            cvAddWeighted(pGradnt, 1, pInterim, -2.0*grayScaleDiff/validNum, 0, pGradnt); 
            // Hessian matrix
            cvGEMM(pInterim, pInterim, -2.0/validNum, 
                    pSecDerv, 1, pSecDerv, CV_GEMM_B_T);
            cvGEMM(pDx2Da,   pDx2Da,   -2.0*grayScaleDiff*img2DxDx/validNum, 
                    pSecDerv, 1, pSecDerv, CV_GEMM_B_T); 
            cvGEMM(pDy2Da,   pDx2Da,   -2.0*grayScaleDiff*img2DxDy/validNum,
                    pSecDerv, 1, pSecDerv, CV_GEMM_B_T);
            cvAddWeighted(pSecDerv, 1, pDx2DaDa, -2.0*grayScaleDiff*img2Dx/validNum, 0, pSecDerv);
            cvGEMM(pDx2Da,   pDy2Da,   -2.0*grayScaleDiff*img2DxDy/validNum,
                    pSecDerv, 1, pSecDerv, CV_GEMM_B_T);
            cvGEMM(pDy2Da,   pDy2Da,   -2.0*grayScaleDiff*img2DyDy/validNum,
                    pSecDerv, 1, pSecDerv, CV_GEMM_B_T);
            cvAddWeighted(pSecDerv, 1, pDy2DaDa, -2.0*grayScaleDiff*img2Dy/validNum, 0, pSecDerv);
            
            *pErr += grayScaleDiff*grayScaleDiff/validNum;
        }
    }

    cvInvert(pSecDerv, pSecDerv, CV_SVD);   
    cvMatMul(pSecDerv, pGradnt,  pNewtonVect);

    cvReleaseMat(&pDx2Da);
    cvReleaseMat(&pDy2Da);
    cvReleaseMat(&pDx2DaDa);
    cvReleaseMat(&pDy2DaDa);
    cvReleaseMat(&pGradnt);
    cvReleaseMat(&pSecDerv);
    cvReleaseMat(&pInterim);

    return 0;
}


void UpdateTransMat(CvMat *pTransMat, const CvMat *pVect, const double step)
{
   int i; 

   // Only first 8 elements need modification
   for (i=0; i<8; i++)
   {
       int r = i/3;
       int c = i%3;

       double elmt;
       elmt =  cvGetReal2D(pTransMat, r, c);
       elmt += -step*cvGetReal1D(pVect, i);

       cvSetReal2D(pTransMat, r, c, elmt);
   }
}

double myFabs(double v)
{
    if (v < 0)
        return -v;
    else
        return v;
}


double ImproveThroughIteration(CvMat *transMat, const IplImage *img1, const IplImage *img2)
{
    CvRect roi;
    CvSize img1Sz, img2Sz;
    
    double mat33 = cvGetReal2D(transMat, 2, 2);
    int    i;
    for (i=0; i<8; i++)
    {
        double elmt = cvGetReal2D(transMat, i/3, i%3);
        cvSetReal2D(transMat, i/3, i%3, elmt/mat33);
    }
    cvSetReal2D(transMat, 2, 2, 1.0);

    img1Sz = cvGetSize(img1);
    img2Sz = cvGetSize(img2);

    IplImage *img1Gray = cvCreateImage(img1Sz, IPL_DEPTH_8U, 1);
    IplImage *img2Gray = cvCreateImage(img2Sz, IPL_DEPTH_8U, 1);
    cvCvtColor(img1, img1Gray, CV_RGB2GRAY);
    cvCvtColor(img2, img2Gray, CV_RGB2GRAY);

    // calculate gradient in x and y direction
    IplImage *img2dx = cvCreateImage(img2Sz, IPL_DEPTH_32F, 1);
    IplImage *img2dy = cvCreateImage(img2Sz, IPL_DEPTH_32F, 1);
    IplImage *img2dxdx = cvCreateImage(img2Sz, IPL_DEPTH_32F, 1);
    IplImage *img2dydy = cvCreateImage(img2Sz, IPL_DEPTH_32F, 1);
    IplImage *img2dxdy = cvCreateImage(img2Sz, IPL_DEPTH_32F, 1);
    cvSobel(img2Gray, img2dx, 1, 0, 3);
    cvSobel(img2Gray, img2dy, 0, 1, 3);
    cvSobel(img2dx,   img2dxdx, 1, 0, 3);
    cvSobel(img2dy,   img2dydy, 0, 1, 3); 
    cvSobel(img2dx,   img2dxdy, 0, 1, 3);

    CvMat *pImprove = cvCreateMat(8, 1, CV_64FC1);

    CvMat *lastTransMat = cvCloneMat(transMat);
    CvMat *lastGradnt   = cvCloneMat(pImprove);
    double err;
    double lastErr  = 0;
    int    interNum = 0;
    double step     = -1.0;
    while(1)
    {
        gpc_polygon intersect;

        /* map img2 to img1, and find the intersection */
        IntersectTwoImages(&intersect, img1, img2, transMat);
        /* Containing rectangle for intersection region */
        FindContainRect(&roi, &intersect, &img1Sz);

        CvMat *projReslt = cvCreateMat(roi.height, roi.width, CV_64FC3);

        int validNum = MapRoi(projReslt, &roi, transMat, &img2Sz);
        
        /* theta(A) = (1/num(roi))*sum((I_1(x_1,y_1)-I_2(x_2,y_2))^2) */
        CalcNewtonVector(pImprove, &err, 
                img1Gray, img2Gray, img2dx, img2dy, 
                img2dxdx, img2dydy, img2dxdy, 
                &roi, projReslt, validNum);

        /* Release dynamicly allocated resource */
        cvReleaseMat(&projReslt);
        gpc_free_polygon(&intersect);

        if (myFabs(err - lastErr) < IMPROVE_VALUE_THRESHOLD || 
            myFabs(err - lastErr)/lastErr < IMPROVE_RATIO_THRESHOLD)
        {
            printf("err=%lf, lastErr=%lf, break\n", err, lastErr);
            break;
        }

        if (interNum > INTER_TIMES_MAX)
        {
            printf("Interation reach max times(%d)", INTER_TIMES_MAX);
            break;
        }

        lastErr = err;
        cvCopy(transMat, lastTransMat, NULL);
        cvCopy(pImprove, lastGradnt, NULL);
        UpdateTransMat(transMat, pImprove, step);

        printf("Iter[%d]: err=%lf\n", interNum, err);

        interNum ++;
    }

    cvCopy(lastTransMat, transMat, NULL);

    cvReleaseMat(&pImprove);
    cvReleaseMat(&lastTransMat);
    cvReleaseImage(&img2dx);
    cvReleaseImage(&img2dy);

    return err;
}


void DrawCross(IplImage *pIplImg, int x, int y, const char *pText)
{
    static CvFont font;
    static int    fontInit = false;

    if (fontInit == false)
    {
        cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.4, 0.4);
        fontInit = true;
    }

    cvLine(pIplImg, cvPoint(x-2, y), cvPoint(x+3, y), cvScalar(0, 0, 255));
    cvLine(pIplImg, cvPoint(x, y-2), cvPoint(x, y+3), cvScalar(0, 0, 255));
    cvPutText(pIplImg, pText, cvPoint(x,y), &font, cvScalar(255,255,255));
}


#define SIFT_TREE_SPLITS 3
#define SIFT_TREE_DEPTH  8 
#define MAX_INPUT_IMGS   200

int main3(int argc, char **argv)
{
    int nimgs, i;
    IplImage       *iplImgs[MAX_INPUT_IMGS];
    struct feature *ppFeat[MAX_INPUT_IMGS];
    int             nFeat[MAX_INPUT_IMGS];

    memset(iplImgs, 0, sizeof(iplImgs));
    memset(ppFeat,  0, sizeof(ppFeat));
    memset(nFeat,   0, sizeof(nFeat));

    SiftTree siftTree(SIFT_TREE_SPLITS, SIFT_TREE_DEPTH);

    nimgs = argc-1;
    for (i=0; i<nimgs; i++)
    {
        iplImgs[i] = cvLoadImage(argv[i+1], 1);

        nFeat[i] = sift_features(iplImgs[i], &ppFeat[i]);
        siftTree.AddSiftFeature(ppFeat[i], nFeat[i], i);
    }

    siftTree.BuildTree();

    int vwMax = ((int)pow(SIFT_TREE_SPLITS, SIFT_TREE_DEPTH+1)-1)/(SIFT_TREE_SPLITS-1);
    std::vector<int> vwCount(vwMax, 0);

    for (i=0; i<nimgs; i++)
    {
        int j;
        for (j=0; j<nFeat[i]; j++)
        {
            unsigned int vw;
            siftTree.Quantize(&vw, ppFeat[i][j].descr);
            ppFeat[i][j].feat_class = vw;
            vwCount[vw] += 1;
        }
    }

    const char *pWinNameTplt = "img%03d";
    char  winNameBuf[20];
    char  times[20];
    for (i=0; i<nimgs; i++)
    {
        int j;
        for (j=0; j<nFeat[i]; j++)
        {
            int vw = ppFeat[i][j].feat_class;

            if (vwCount[vw] > 1)
            {
                snprintf(times, sizeof(times), "%d", vw);
                DrawCross(iplImgs[i], ppFeat[i][j].x, ppFeat[i][j].y, times);
            }
        }

        snprintf(winNameBuf, sizeof(winNameBuf), pWinNameTplt, i);
        cvNamedWindow(winNameBuf, CV_WINDOW_AUTOSIZE);
        cvShowImage(winNameBuf, iplImgs[i]);
    }
    
    cvWaitKey();

    // Release resource
    for (i=0; i<argc-1; i++)
        cvReleaseImage(&iplImgs[i]);

    return 0;
}


int main2( int argc, char** argv )
{

    IplImage  *img1;
    IplImage  *img2;

    struct feature* feat1, *feat2;
    struct kd_node* kd_root;
    int n1, n2, i;

    if (argc != 3)
    {
        printf("Error! Format: bin image_file1 image_file2");
        return 0;
    }

    img1_file = argv[1];
    img2_file = argv[2];

    CvMemStorage* memstorage = cvCreateMemStorage(0);
    CvSeq *up_seq = cvCreateSeq(    CV_SEQ_ELTYPE_POINT,
                    sizeof(CvSeq),
                    sizeof(CvPoint),
                    memstorage); 
    CvSeq *down_seq = cvCreateSeq(    CV_SEQ_ELTYPE_POINT,
                    sizeof(CvSeq),
                    sizeof(CvPoint),
                    memstorage);

    img1 = cvLoadImage( img1_file, 1 );
    if( ! img1 )
        fatal_error( "unable to load image from %s", img1_file );
    img2 = cvLoadImage( img2_file, 1 );
    if( ! img2 )
        fatal_error( "unable to load image from %s", img2_file );

    stacked = stack_imgs( img1, img2 );


    fprintf( stderr, "Finding features in %s...\n", img1_file );
    n1 = sift_features( img1, &feat1 );


    fprintf( stderr, "Finding features in %s...\n", img2_file );
    n2 = sift_features( img2, &feat2 );

    if(1)
    {
        //draw_features( img1, feat1, n1 );
        cvNamedWindow( "img1", 1 );
        cvShowImage( "img1", img1 );

    }

    if(1)
    {
        //draw_features( img2, feat2, n2 );
        cvNamedWindow( "img2", 1 );
        cvShowImage( "img2", img2 );
    }

    MatchSiftFeats(&kd_root, feat1, n1, feat2, n2, up_seq, down_seq, img1->height);

    CvMat *transMat = 0;  // perspective transformation matrix
    struct feature **inliers;
    int    nin;

    transMat = ransac_xform(feat1, n1, FEATURE_FWD_MATCH, 
                            lsq_homog, 4, 0.5, homog_xfer_err, 3.0, 
                            &inliers, &nin);
    if (nin == 0)
    {
        printf("Sorry. No homography is found!\n");
        return 0;
    }

    resize_img();

    ImproveThroughIteration(transMat, img1, img2);

    CvPoint   nc[4];
    CvPoint   upLeft, downRight;
    CvSize    sz;
    CvPoint   shift1, shift2;
    IplImage  *transImg;

    nc[0] = ProjPoint(cvPoint(0, 0), transMat);
    nc[1] = ProjPoint(cvPoint(img1->width-1, 0), transMat);
    nc[2] = ProjPoint(cvPoint(img1->width-1, img1->height-1), transMat);
    nc[3] = ProjPoint(cvPoint(0, img1->height-1), transMat);

    upLeft.x    = nc[0].x;
    upLeft.y    = nc[0].y;
    downRight.x = nc[0].x;
    downRight.y = nc[0].y;
    for (i=1; i<4; i++)
    {
        if (nc[i].x < upLeft.x)
        {
            upLeft.x = nc[i].x;
        }
        if (nc[i].x > downRight.x)
        {
            downRight.x = nc[i].x;
        }

        if (nc[i].y < upLeft.y)
        {
            upLeft.y = nc[i].y;
        }
        if (nc[i].y > downRight.y)
        {
            downRight.y = nc[i].y;
        }
    }

    sz.height  = downRight.y - upLeft.y + 1;
    sz.width   = downRight.x - upLeft.x + 1;
    shift1.x   = - upLeft.x;
    shift1.y   = - upLeft.y;

    transImg   = cvCreateImage(sz, img1->depth, img1->nChannels);

    // Transform img1 to align with img2, then shift it
    warp_by_matrix<unsigned char>(img1, transImg, transMat, shift1);

    IplImage  *imagePart1;
    IplImage  *imagePart2;
    IplImage  *stitchedImage;
    IplImage  *stitchedImage2;
    CvPoint   shift_t;

    // Calculate the new border
    nc[0].x = 0;
    nc[0].y = 0;
    nc[1].x = img2->width - 1;
    nc[1].y = 0;
    nc[2].x = img2->width - 1;
    nc[2].y = img2->height - 1;
    nc[3].x = 0;
    nc[3].y = img2->height - 1;
    for (i=1; i<4; i++)
    {
        if (nc[i].x < upLeft.x)
        {
            upLeft.x = nc[i].x;
        }
        if (nc[i].x > downRight.x)
        {
            downRight.x = nc[i].x;
        }

        if (nc[i].y < upLeft.y)
        {
            upLeft.y = nc[i].y;
        }
        if (nc[i].y > downRight.y)
        {
            downRight.y = nc[i].y;
        }
    }

    sz.width  = (((downRight.x - upLeft.x + 1) + 3) & ~3);
    sz.height = (((downRight.y - upLeft.y + 1) + 3) & ~3); 
    
    shift2.x = - upLeft.x;
    shift2.y = - upLeft.y;

    stitchedImage   = cvCreateImage(sz, img1->depth, img1->nChannels);
    stitchedImage2  = cvCreateImage(sz, img1->depth, img1->nChannels);
    imagePart1      = cvCreateImage(sz, img1->depth, img1->nChannels);
    imagePart2      = cvCreateImage(sz, img1->depth, img1->nChannels);

    shift_t.x = -shift1.x + shift2.x;
    shift_t.y = -shift1.y + shift2.y;

    // Map the first image
    // Because transImg is already shifted by shift1 to make sure that there are no negative image coordinates,
    // so we have to shift it back by shift1, then shift it by shift2 which is used to make sure there are no
    // negative image coordinates in the new stitchedImg
    shift_image<unsigned char>(transImg, imagePart1, shift_t);

    shift_t.x = shift2.x;
    shift_t.y = shift2.y;
    shift_image<unsigned char>(img2, imagePart2, shift_t);

    OverlayImages(imagePart1, 0.5, imagePart2, 0.5, stitchedImage);
    cvNamedWindow("Stitched");
    cvShowImage("Stitched", stitchedImage);
    cvSaveImage("stitched.jpg", stitchedImage);

    cvWaitKey(0);

    //------�ͷ��ڴ��洢��-----------
    cvReleaseMemStorage( &memstorage );
    cvReleaseImage( &stacked );
    cvReleaseImage( &img1 );
    cvReleaseImage( &img2 );
    kdtree_release( kd_root );
    free( feat1 );
    free( feat2 );
    return 0;
}



void on_mouse( int event, int x, int y, int flags, void* param ) 
{
    if( (event==CV_EVENT_LBUTTONUP) &&  (flags==CV_EVENT_FLAG_CTRLKEY) ) 
    {
                
    }
    
    if( (event==CV_EVENT_LBUTTONUP) &&  (flags==CV_EVENT_FLAG_ALTKEY) ) 
        //ALT������������ʼ���µ�ʱ���Ŵ�ƥ������ͼ
    {
        
        
        //---------������ƥ�亯��--------------    
        if(imgzoom_scale<1.5)
        {
            imgzoom_scale=1.1*imgzoom_scale;
        }
        
        else imgzoom_scale=1.0;
        //----�Ŵ�ƥ������ͼ-----
        resize_img();

    }
    
    if( (event==CV_EVENT_RBUTTONUP) &&  (flags==CV_EVENT_FLAG_ALTKEY) ) 
        
        //ALT�������Ҽ���ʼ���µ�ʱ����Сƥ������ͼ
    {
        
        
        //---------������ƥ�亯��--------------    
        if(imgzoom_scale>0.0)
        {
            imgzoom_scale=0.9*imgzoom_scale;
        }
        
        else imgzoom_scale=0.5;

        //----��Сƥ������ͼ-----
        resize_img();

        //printf("%f\n",imgzoom_scale);

    }
    

}



//--------------�ı�ƥ������ͼ���Ĵ�С------------------------

void resize_img()
{
    IplImage* resize_stacked;

    resize_stacked=cvCreateImage(cvSize(  (int)(stacked->width*imgzoom_scale),  (int)(stacked->height*imgzoom_scale)  ),
                stacked->depth,
                stacked->nChannels);
    //----����ƥ������ͼ�Ĵ�С------

    cvResize(stacked, resize_stacked, CV_INTER_AREA );


    cvNamedWindow( "Matches", 1 );

    
    //---------������Ӧ����--------------
    
    cvSetMouseCallback("Matches", on_mouse, 0 );
    

    cvShowImage( "Matches", resize_stacked);

    //cvWaitKey( 0 );
}


void OverlayImages(const IplImage *part1, double w1, const IplImage *part2, double w2, IplImage *out)
{
    unsigned char *pSrc1, *pSrc2;
    unsigned char *pDst;
    int           i, j;

    assert(part1->width == part2->width);
    assert(part1->height == part2->height);
    assert(part1->depth == part2->depth);
    assert(part1->nChannels == part2->nChannels);

    i = 0;
    while (i < part1->height)
    {
        pSrc1   = (unsigned char *)(part1->imageData + i * part1->widthStep);
        pSrc2   = (unsigned char *)(part2->imageData + i * part2->widthStep);
        pDst    = (unsigned char *)(out->imageData + i * out->widthStep);

        for (j=0; j<out->widthStep; j++)
        {
            double tmp;

            if (pSrc1[j] != 0 && pSrc2[j] != 0)
                tmp = pSrc1[j] * w1 + pSrc2[j] * w2;
            else
                tmp = (pSrc1[j])?(pSrc1[j]):(pSrc2[j]);

            if (tmp > 255)
            {
                tmp = 255;
                //print("Value is clamped!\n");
            }
            pDst[j] = (unsigned char)tmp;
        }

        i ++;
    }
}

/**
 * Blend images using algorithm proposed by Burt and Adelson. Please refer to 
 * "A Multiresolution Spline With Application to Image Mosaics" for more info.
 */
template <typename ElementType>
void blend_images(const IplImage *part1, const IplImage *part2, const IplImage *map, IplImage *out)
{
    CvSize     sz;
    int        i, j, k;

    ElementType *ptr_1, *ptr_2, *ptr_out;
    unsigned char *ptr_map;

    assert(part1->width == part2->width);
    assert(part1->height == part2->height);
    assert(part1->width == out->width);
    assert(part1->height == out->height);
    assert(part1->width == map->width);
    assert(part1->height == map->height);
    assert((part1->depth&0xff) == (8*sizeof(ElementType)));
    assert((map->depth&0xff) == 8);

    sz = cvGetSize(part1);

    // Synthesize the low freq band
    for (i=0; i<sz.height; i++)
    {
        ptr_1   = (ElementType *)(part1->imageData + i*part1->widthStep);
        ptr_2   = (ElementType *)(part2->imageData + i*part2->widthStep);
        ptr_out = (ElementType *)(out->imageData + i*out->widthStep);
        ptr_map = (unsigned char *)(map->imageData + i*map->widthStep);

        for (j=0; j<sz.width; j++)
        {
            for (k=0; k<part1->nChannels; k++)
                ptr_out[k] = (ElementType)((ptr_map[0]*ptr_1[k] + (int)(255-ptr_map[0])*ptr_2[k]) / 255);

            ptr_1   += part1->nChannels;
            ptr_2   += part2->nChannels;
            ptr_out += out->nChannels;
            ptr_map += 1;
        }
    }
}

CvPoint ProjPoint(CvPoint in, CvMat *trans)
{
    double   tx, ty, t;
    double   *db;
    CvPoint  res;
    
    int depth, cn;
    depth = CV_MAT_DEPTH(trans->type);
    cn    = CV_MAT_CN(trans->type);
    assert(depth == CV_64F);
    assert(cn == 1);
    db    = trans->data.db;

    tx = db[0] * in.x + db[1] * in.y + db[2];
    ty = db[3] * in.x + db[4] * in.y + db[5];
    t  = db[6] * in.x + db[7] * in.y + db[8];

    res.x = (int)(tx / t + 0.5);
    res.y = (int)(ty / t + 0.5);

    return res;
}

/**
 * Warp by a transformation matrix. src(x', y') = dst(x, y), and t*[x', y', 1]' = transMat * [x, y, 1]'
 */
template <typename ElementType>
void warp_by_matrix   (IplImage *src, IplImage *dst, CvMat *transMat, CvPoint shift)
{
    CvSize sz;
    double *db;
    int    i, j, k;
    double t, tx, ty;
    double xl, xr, yu, yd;
    int    srcx, srcy;
    double tmp;
    int    s;

    gsl_matrix      *mat;
    gsl_matrix      *invm;
    gsl_permutation *perm;

    ElementType *pSrc11, *pSrc12;
    ElementType *pSrc21, *pSrc22;
    ElementType *pDst;
    
    assert(CV_MAT_DEPTH(transMat->type) == CV_64F);
    assert(CV_MAT_CN(transMat->type) == 1);
    assert(transMat->rows == transMat->cols);
    assert(transMat->rows == 3);
    
    db    = transMat->data.db;
    mat   = gsl_matrix_alloc(3, 3);
    invm  = gsl_matrix_alloc(3, 3);
    perm  = gsl_permutation_alloc(3);
    assert(mat->tda == 3);
    memcpy(mat->data, db, sizeof(double)*9);

    gsl_linalg_LU_decomp(mat, perm, &s);
    gsl_linalg_LU_invert(mat, perm, invm);
    memcpy(db,invm->data, sizeof(double)*9);

    cvZero(dst);
    sz = cvGetSize(dst);
    for (i=0; i<sz.height; i++) {
        for (j=0; j<sz.width; j++) {
            tx = db[0]*(j-shift.x) + db[1]*(i-shift.y) + db[2];
            ty = db[3]*(j-shift.x) + db[4]*(i-shift.y) + db[5];
            t  = db[6]*(j-shift.x) + db[7]*(i-shift.y) + db[8];

            tx /= t;
            ty /= t;
            
            if (tx >= 0 && tx < (src->width-1)) {
                srcx = (int)tx;
                xl   = tx - srcx;
                xr   = 1.0 - xl;
            } else if (tx == (src->width-1)) {
                srcx = (int)(tx - 1.0);
                xl   = 1.0;
                xr   = 0.0;
            } else {
#if COMPLAIN_OUT_OF_BOUND
                printf("x-coordinate %lf out of bound\n", tx);
#endif
                continue;
            }

            if (ty >= 0 && ty < (src->height-1)) {
                srcy = (int)ty;
                yu   = ty - srcy;
                yd   = 1.0 - yu;
            } else if (ty == (src->height-1)) {
                srcy = (int)(ty - 1.0);
                yu   = 1.0;
                yd   = 0.0;
            } else {
#if COMPLAIN_OUT_OF_BOUND
                printf("y-coordinate %lf out of bound\n", ty);
#endif
                continue;
            }

            pSrc11 = (ElementType *)(src->imageData + src->widthStep * srcy + src->nChannels * srcx);
            pSrc12 = pSrc11 + src->nChannels;
            pSrc21 = (ElementType *)((char *)pSrc11 + src->widthStep);
            pSrc22 = pSrc21 + src->nChannels;
            pDst   = (ElementType *)(dst->imageData + dst->widthStep * i + dst->nChannels * j);

            for (k=0; k<src->nChannels; k++) {
                tmp     = (*pSrc11)*yd*xr + (*pSrc21)*yu*xr + (*pSrc12)*yd*xl + (*pSrc22)*yu*xl;
                (*pDst) = (ElementType)(tmp);

                pSrc11 ++;
                pSrc12 ++;
                pSrc21 ++;
                pSrc22 ++;
                pDst   ++;
            }
        }
    }

    gsl_permutation_free(perm);
    gsl_matrix_free(invm);
    gsl_matrix_free(mat);
}

/**
 * Shift source image to fit into the dst image.
 */
template <typename ElementType>
void shift_image(const IplImage *src, IplImage *dst, CvPoint shift)
{
    int i, j, k;
    ElementType *pSrc;
    ElementType *pDst;

    assert(src->nChannels == dst->nChannels);
    assert(src->depth == dst->depth);
    assert((src->depth & 0xff) == 8*sizeof(ElementType));

    cvZero(dst);

    for (i=0; i<src->height; i++)
    {
        if ((i + shift.y) < 0 || (i + shift.y) >= dst->height)
            continue;

        pSrc = (ElementType *)(src->imageData + i * src->widthStep);
        pDst = (ElementType *)(dst->imageData + (i+shift.y) * dst->widthStep); 

        for (j=0; j<src->width; j++)
        {
            if ((j + shift.x) < 0 || (j + shift.x) >= dst->width)
                continue;
            
            for (k=0; k<src->nChannels; k++)
            {
                pDst[dst->nChannels * (j+shift.x) + k] = pSrc[src->nChannels * j + k];
            }
        }
    }
}

//static struct feature **inliers = 0;

///**
// * The match error [xerr_i, yerr_i]^T
// */
//void calc_residual(double *p, double *x, int m, int n, void *data)
//{
//    int i;
//    double x1, y1; // Points to be matched by transfrom (x2, y2)
//    double x2, y2;
//    double den;
//
//    for (i=0; i<n/2; i++)
//    {
//        x2 = inliers[i]->x;
//        y2 = inliers[i]->y;
//        x1 = inliers[i]->fwd_match->x;
//        y1 = inliers[i]->fwd_match->y;
//
//        den = p[6]*x2 + p[7]*y2 + 1;
//        x[2*i]   = x1 - (p[0]*x2 + p[1]*y2 + p[2])/den;
//        x[2*i+1] = y1 - (p[3]*x2 + p[4]*y2 + p[5])/den;
//    }
//}
//
///**
// * The jacobian matrix of the match error
// */
//void jac_residual(double *p, double *jac, int m, int n, void *data)
//{
//    int i, j;
//    double x1, y1;
//    double x2, y2;
//    double den, numa, numb;
//
//    j = 0;
//    for (i=0; i<n/2; i++)
//    {
//        x2 = inliers[i]->x;
//        y2 = inliers[i]->y;
//        x1 = inliers[i]->fwd_match->x;
//        y1 = inliers[i]->fwd_match->y;
//
//        numa = p[0]*x2 + p[1]*y2 + p[2];
//        numb = p[3]*x2 + p[4]*y2 + p[5];
//        den  = p[6]*x2 + p[7]*y2 + 1.0;
//
//        /* row 2*i */
//
//        jac[j++] = - x2/den;
//        jac[j++] = - y2/den;
//        jac[j++] = - 1/den;
//
//        jac[j++] = 0;
//        jac[j++] = 0;
//        jac[j++] = 0;
//
//        jac[j++] = - numa * x2 / (den * den);
//        jac[j++] = - numa * y2 / (den * den);
//
//        /* row 2*i+1 */
//
//        jac[j++] = 0;
//        jac[j++] = 0;
//        jac[j++] = 0;
//
//        jac[j++] = - x2/den;
//        jac[j++] = - y2/den;
//        jac[j++] = - 1/den;
//
//        jac[j++] = - numb * x2 / (den * den);
//        jac[j++] = - numb * y2 / (den * den);
//    }
//}

///**
// * Given an initial transform matrix estimate and feature point pairs, 
// * then use leverberg-marquet algorithm to optimize the result.
// */
//void nonlinear_optimize(CvMat *trans, struct feature **in, int nin)
//{
//    double params[9];
//    double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
//    double *residual;
//    double ret;
//    int    i;
//
//    inliers = in;
//
//    opts[0]=LM_INIT_MU; 
//    opts[1]=1E-15; 
//    opts[2]=1E-15; 
//    opts[3]=1E-20;
//    opts[4]=LM_DIFF_DELTA; // relevant only if the finite difference Jacobian version is used
//    
//    int depth, cn;
//    depth = CV_MAT_DEPTH(trans->type);
//    cn    = CV_MAT_CN(trans->type);
//    assert(depth == CV_64F);
//    assert(cn == 1);
//
//    memcpy(params, trans->data.db, sizeof(double)*9);
//    residual = (double *)malloc(sizeof(double)*2*nin);
//
//    // Calculate residual using given parameters 
//    memset(residual, 0, sizeof(double)*2*nin);
//
//    ret = dlevmar_der(calc_residual, jac_residual, params, residual, 8, 2*nin, 1000, opts, info, NULL, NULL, NULL);
//    return;
//}
