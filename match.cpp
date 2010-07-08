#include "sift.h"
#include "imgfeatures.h"
#include "kdtree.h"
#include "utils.h"
#include "xform.h"
#include "gpc.h"

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

#ifdef _DEBUG
#define COMPLAIN_OUT_OF_BOUND 0
#endif

/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

/* threshold on squared ratio of distances between NN and 2nd NN */
#define NN_SQ_DIST_RATIO_THR 0.49

#define EQUAL_IMAGE_SIZE(src, dst) (((src)->width == (dst)->width)&&((dst)->height == (dst)->height))

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

    memset(&invImg2Bound, 0, sizeof(gpc_polygon));
    memset(&img1Bound,    0, sizeof(gpc_polygon));
    memset(&intersect,    0, sizeof(gpc_polygon));

    gpc_vertex_list img1VertLst, img2VertLst;
    img1VertLst.num_vertices = 4;
    img1VertLst.vertex       = (gpc_vertex *)calloc(4, sizeof(gpc_vertex));
    img2VertLst.num_vertices = 4;
    img2VertLst.vertex       = (gpc_vertex *)calloc(4, sizeof(gpc_vertex));

    img1VertLst.vertex[0].x = 0;
    img1VertLst.vertex[0].y = 0;
    img1VertLst.vertex[1].x = img1->width;
    img1VertLst.vertex[1].y = 0;
    img1VertLst.vertex[2].x = img1->width;
    img1VertLst.vertex[2].y = img1->height;
    img1VertLst.vertex[3].x = 0;
    img1VertLst.vertex[3].y = img1->height;

    CvPoint img2Corners[4];
    img2Corners[0].x = 0;
    img2Corners[0].y = 0;
    img2Corners[1].x = img2->width;
    img2Corners[1].y = 0;
    img2Corners[2].x = img2->width;
    img2Corners[2].y = img2->height;
    img2Corners[3].x = 0;
    img2Corners[3].y = img2->height;

    // Convert img2 corners use "invTransMat x [corner.x, corner.y, 1]^T"
    int i;
    for (i=0; i<4; i++)
    {
        cvSetReal1D(tmpVector, 0, img2Corners[i].x);
        cvSetReal1D(tmpVector, 1, img2Corners[i].y);
        cvSetReal1D(tmpVector, 2, 1.0);

        cvMatMul(invTransMat, tmpVector, resltVector);

        double scale = cvGetReal1D(resltVector, 2);
        cvScale(resltVector, resltVector, 1.0/scale);

        img2VertLst.vertex[i].x = cvGetReal1D(resltVector, 0);
        img2VertLst.vertex[i].y = cvGetReal1D(resltVector, 1);
    }

    gpc_add_contour(&invImg2Bound, &img2VertLst, 0);
    gpc_add_contour(&img1Bound,    &img1VertLst, 0);

    gpc_polygon_clip(GPC_INT, &img1Bound, &invImg2Bound, intersect);

    cvReleaseMat(&invTransMat);
    cvReleaseMat(&tmpVector);
    cvReleaseMat(&resltVector);
    free(img1VertLst.vertex);
    free(img2VertLst.vertex);

    return 0;
}


int main( int argc, char** argv )
{

	IplImage  *img1;
	IplImage  *img2;

	struct feature* feat1, *feat2;
	struct kd_node* kd_root;
	int n1, n2, i;
    int ret;

	if (argc != 3)
	{
		printf("Error! Format: bin image_file1 image_file2");
		return 0;
	}

	img1_file = argv[1];
	img2_file = argv[2];

	CvMemStorage* memstorage = cvCreateMemStorage(0);
	CvSeq *up_seq = cvCreateSeq(	CV_SEQ_ELTYPE_POINT,
					sizeof(CvSeq),
					sizeof(CvPoint),
					memstorage); 
	CvSeq *down_seq = cvCreateSeq(	CV_SEQ_ELTYPE_POINT,
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

	transMat = ransac_xform(feat1, n1, FEATURE_FWD_MATCH, lsq_homog, 4, 0.5, homog_xfer_err, 3.0, &inliers, &nin);
	resize_img();

    //cvSaveImage("matchedfeat.jpg", stacked);
    if (nin == 0)
    {
        printf("Sorry. No homography is found!\n");
        return 0;
    }

    gpc_polygon intersect;
    IntersectTwoImages(&intersect, img1, img2, transMat);

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

	cvWaitKey( 0 );
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
//	int i;
//	double x1, y1; // Points to be matched by transfrom (x2, y2)
//	double x2, y2;
//	double den;
//
//	for (i=0; i<n/2; i++)
//	{
//		x2 = inliers[i]->x;
//		y2 = inliers[i]->y;
//		x1 = inliers[i]->fwd_match->x;
//		y1 = inliers[i]->fwd_match->y;
//
//		den = p[6]*x2 + p[7]*y2 + 1;
//		x[2*i]   = x1 - (p[0]*x2 + p[1]*y2 + p[2])/den;
//		x[2*i+1] = y1 - (p[3]*x2 + p[4]*y2 + p[5])/den;
//	}
//}
//
///**
// * The jacobian matrix of the match error
// */
//void jac_residual(double *p, double *jac, int m, int n, void *data)
//{
//	int i, j;
//	double x1, y1;
//	double x2, y2;
//	double den, numa, numb;
//
//	j = 0;
//	for (i=0; i<n/2; i++)
//	{
//		x2 = inliers[i]->x;
//		y2 = inliers[i]->y;
//		x1 = inliers[i]->fwd_match->x;
//		y1 = inliers[i]->fwd_match->y;
//
//		numa = p[0]*x2 + p[1]*y2 + p[2];
//		numb = p[3]*x2 + p[4]*y2 + p[5];
//		den  = p[6]*x2 + p[7]*y2 + 1.0;
//
//		/* row 2*i */
//
//		jac[j++] = - x2/den;
//		jac[j++] = - y2/den;
//		jac[j++] = - 1/den;
//
//		jac[j++] = 0;
//		jac[j++] = 0;
//		jac[j++] = 0;
//
//		jac[j++] = - numa * x2 / (den * den);
//		jac[j++] = - numa * y2 / (den * den);
//
//		/* row 2*i+1 */
//
//		jac[j++] = 0;
//		jac[j++] = 0;
//		jac[j++] = 0;
//
//		jac[j++] = - x2/den;
//		jac[j++] = - y2/den;
//		jac[j++] = - 1/den;
//
//		jac[j++] = - numb * x2 / (den * den);
//		jac[j++] = - numb * y2 / (den * den);
//	}
//}

///**
// * Given an initial transform matrix estimate and feature point pairs, 
// * then use leverberg-marquet algorithm to optimize the result.
// */
//void nonlinear_optimize(CvMat *trans, struct feature **in, int nin)
//{
//	double params[9];
//	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
//	double *residual;
//	double ret;
//	int    i;
//
//	inliers = in;
//
//	opts[0]=LM_INIT_MU; 
//	opts[1]=1E-15; 
//	opts[2]=1E-15; 
//	opts[3]=1E-20;
//	opts[4]=LM_DIFF_DELTA; // relevant only if the finite difference Jacobian version is used
//	
//	int depth, cn;
//	depth = CV_MAT_DEPTH(trans->type);
//	cn    = CV_MAT_CN(trans->type);
//	assert(depth == CV_64F);
//	assert(cn == 1);
//
//	memcpy(params, trans->data.db, sizeof(double)*9);
//	residual = (double *)malloc(sizeof(double)*2*nin);
//
//	// Calculate residual using given parameters 
//	memset(residual, 0, sizeof(double)*2*nin);
//
//	ret = dlevmar_der(calc_residual, jac_residual, params, residual, 8, 2*nin, 1000, opts, info, NULL, NULL, NULL);
//	return;
//}
