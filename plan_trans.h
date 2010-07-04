#ifndef _PLAN_TRANS_H_
#define _PLAN_TRANS_H_

#include <cxcore.h>
#include <stdio.h>
#include <vector>

/**
* Solve for the parameters control the affine transformation including shift, rotation, shrinking and enlarging in a plane.
*               x2           |k*cos(theta)  -k*sin(theta)  xs|    x1
*               y2      =    |k*sin(theta)   k*cos(theta0  ys|  * y1
*               1            |0              0              1|     1
* Parameters are params[0] = k*cos(theta), params[1] = k*sin(theta), params[2] = xs, params[3] = ys.
*
* @param points1 points before transformation
*
* @param points2 points after transfromation, points2[i] is the matching point for points1[i]
*/
double solve_plane_trans_params(std::vector<CvPoint *> points1, std::vector<CvPoint *> points2, double params[]);

/**
 * Given two OpenCV sequences, use LMedS to calculate the affine transformation .
 *
 * @param pt_lst1 Point list one
 *
 * @param pt_lst2 Point list two which is coressponding to the list one
 */
double solve_affine_transform(const CvSeq *pt_lst1, const CvSeq *pt_lst2, double params[4]);

/**
 * Get err of the transformation on the point pair set of point seq one and point seq two.
 *
 * @param pt_seq1 Point sequence one
 *
 * @param pt_seq2 Point sequence two
 */
double get_trans_error(const CvSeq *pt_seq1, const CvSeq *pt_seq2, double params[4]);

#endif
