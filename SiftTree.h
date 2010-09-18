#ifndef __SIFT_TREE_H__
#define __SIFT_TREE_H__

#include "imgfeatures.h"
#include <vector>

struct SiftFeature
{
    double feat[FEATURE_MAX_D];
    int    imgIdx;
};

struct SiftTreeNode
{
    double center[FEATURE_MAX_D];
    int    numChld;
};

typedef std::vector<SiftFeature>   SiftFeatureArray;
typedef std::vector<SiftTreeNode>  SiftNodeArray;
typedef std::vector<SiftFeature>::iterator  SiftFeatureArrayInter;
typedef std::vector<SiftTreeNode>::iterator SiftNodeArrayInter;
typedef std::vector<int>::iterator          IntVectInter;

class SiftTree
{
    public:
    SiftTree(){};
    SiftTree(int nSplits, int maxLvl);
    ~SiftTree(){};

    int AddSiftFeature(const struct feature *pFeat, int nFeat, int nImgIdx);
    int BuildTree();
    void Quantize(unsigned int *pFeatIdx, const double *sift);

    private:
    int KMeansCluster(int level, int splitIdx, int start, int end);

    SiftFeatureArray featArray;
    SiftNodeArray    treeNode;
    std::vector<int> featIdxArr;
    std::vector<int> featClust;
    int maxLevel;
    int nsplits;
};

#endif
