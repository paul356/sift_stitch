#include "SiftTree.h"
#include "imgfeatures.h"

using namespace std;

SiftTree::SiftTree(int nSplits, int maxLvl)
{
    nsplits  = nSplits;
    maxLevel = maxLvl;
}


int SiftTree::AddSiftFeature(const struct feature *pFeat, int nFeat, int nImgIdx)
{
    SiftFeature siftFeat;
    int i;

    for (i=0; i<nFeat; i++)
    {
        memcpy(siftFeat.feat, pFeat->descr, sizeof(siftFeat.feat));
        siftFeat.imgIdx = nImgIdx;

        featArray.push_back(siftFeat);

        pFeat += 1;
    }

    return 0;
}


inline double FeatDist(const double feat1[FEATURE_MAX_D], const double feat2[FEATURE_MAX_D])
{
    int i;
    double distt2 = 0.;

    for (i=0; i<FEATURE_MAX_D; i++)
    {
        distt2 += (feat1[i] - feat2[i])*(feat1[i] - feat2[i]);
    }

    return distt2;
}


inline void ArrangeFeatIndex(vector<int> *pFeatClust, 
        vector<int> *pFeatIdxArr, int featArrStart, int featArrEnd)
{
    int sortBeg, sortEnd;
    vector<int> sortStack;
    
    sortBeg = featArrStart;
    sortEnd = featArrEnd;
    sortStack.push_back(sortBeg);
    sortStack.push_back(sortEnd);

    while (!sortStack.empty())
    {
        sortEnd = sortStack.back();
        sortStack.pop_back();
        sortBeg = sortStack.back();
        sortStack.pop_back();

        if (sortBeg+1 == sortEnd)
            continue;

        int pivot, pivotFeat;
        int p, q;
        pivot     = (*pFeatClust)[sortBeg];
        pivotFeat = (*pFeatIdxArr)[sortBeg];
        p = sortBeg;
        q = sortBeg + 1;
        while (q < sortEnd)
        {
            if ((*pFeatClust)[q] < pivot)
            {
                (*pFeatClust)[p] = (*pFeatClust)[q];
                (*pFeatClust)[q] = (*pFeatClust)[p+1];
                (*pFeatIdxArr)[p] = (*pFeatIdxArr)[q];
                (*pFeatIdxArr)[q] = (*pFeatIdxArr)[p+1];
                p += 1;
            }
            q ++;
        }

        (*pFeatClust)[p]  = pivot;
        (*pFeatIdxArr)[p] = pivotFeat;

        if (p > sortBeg)
        {
            sortStack.push_back(sortBeg);
            sortStack.push_back(p);
        }
        if (sortEnd > p+1)
        {
            sortStack.push_back(p+1);
            sortStack.push_back(sortEnd);
        }
    }
}


int SiftTree::KMeansCluster(int level, int chldStart, int start, int end)
{
    int featNum;
    vector<SiftTreeNode> clusts(nsplits);

    if (start+nsplits >= end)
    {
        int i=0;
        int inter;
        for (inter=start; inter<end; inter++)
        {
            int idx = featIdxArr[inter];
            memcpy(treeNode[chldStart+i].center, 
                    featArray[idx].feat, sizeof(treeNode[chldStart+i].center));
            treeNode[chldStart+i].numChld = 1;
            i ++;
        }

        //printf("Feat number less than nsplits\n");
        return 0;
    }

    featNum = end - start;

    // 随机初始化初始聚类
    // srand(time(NULL));
    int i, icluster = 0;
    for (i=0; i<featNum; i++)
    {
        int  idx = featIdxArr[start + i];
        bool newFeat = true;
        int j;
        for (j=0; j<icluster; j++)
        {
            if (memcmp(featArray[idx].feat, 
                        clusts[j].center, 
                        sizeof(featArray[idx].feat)) == 0)
            {
                newFeat = false;
                break;
            }
        }

        if (newFeat == true)
        {
            memcpy(clusts[icluster].center, 
                    featArray[idx].feat, sizeof(clusts[icluster].center));
            icluster += 1;

            if (icluster == nsplits)
                break;
        }
    }

    if (icluster < nsplits)
    {
        int i=0;
        for (i=0; i<icluster; i++)
        {
            treeNode[chldStart+i] = clusts[i];
            treeNode[chldStart+i].numChld = 1;
            i ++;
        }

        //printf("Distinct features less than nsplits\n");
        return 0;
    }

    while(1)
    {
        // 初始化类别中子孙数
        for (i=0; i<nsplits; i++)
            clusts[i].numChld = 0;  

        // 重新分配各特征点
        int nchg = 0;
        int inter;
        for (inter=start; inter!=end; inter++)
        {
            int featIdx = featIdxArr[inter];
            double minDist = FeatDist(featArray[featIdx].feat, clusts[0].center);
            int    minIdx  = 0;
            int    j;
            for (j=1; j<nsplits; j++)
            {
                double tmpDist = FeatDist(featArray[featIdx].feat, clusts[j].center);
                if (tmpDist < minDist)
                {
                    minDist = tmpDist;
                    minIdx  = j;
                }
            }

            if (featClust[inter] != minIdx)
            {
                featClust[inter] = minIdx;
                nchg ++;
            }

            clusts[minIdx].numChld += 1;
        }

        // Check convergence
        if (nchg == 0)
        {
            break;
        }

        // Re-caculate the cluster center
        for (i=0; i<nsplits; i++)
            memset(clusts[i].center, 0, sizeof(clusts[i].center));  

        for (inter=start; inter!=end; inter++)
        {
            int clustIdx;

            clustIdx = featClust[inter];
            int j;
            int nd = sizeof(clusts[clustIdx].center)/sizeof(clusts[clustIdx].center[0]);
            for (j=0; j<nd; j++)
            {
                clusts[clustIdx].center[j] 
                    += featArray[featIdxArr[inter]].feat[j] / (double)(clusts[clustIdx].numChld);
            }
        }
    }

    for (i=0; i<nsplits; i++)
    {
        treeNode[chldStart+i] = clusts[i];
    }

    ArrangeFeatIndex(&featClust, &featIdxArr, start, end);

    /** debug purpose */
    static FILE *fdump = NULL;
    if (fdump == NULL)
    {
        fdump = fopen("treedump.txt", "w");
    }
    fprintf(fdump, "[%d](%d){\n", level, featNum);
    for (i=0; i<nsplits; i++)
    {
        fprintf(fdump, "%d\n", clusts[i].numChld);
    }
    //int inter;
    //for (inter=start; inter!=end; inter++)
    //{
        //fprintf(fdump, "%d,", featIdxArr[inter]);
    //}
    //fprintf(fdump, "\n");
    //for (inter=start; inter!=end; inter++)
    //{
        //fprintf(fdump, "%d,", featClust[inter]);
    //}
    //fprintf(fdump, "\n");
    fprintf(fdump, "}\n");
    /** debug purpose */

    if (level < maxLevel) 
    {
        int j;
        int begPos, endPos;
        begPos = start;
        for (j=0; j<nsplits; j++)
        {
            endPos = begPos + clusts[j].numChld;
            KMeansCluster(level+1, (chldStart+j)*nsplits+1, begPos, endPos);
            begPos = endPos;
        }
    }

    return 0;
}


int SiftTree::BuildTree()
{
    int sz = featArray.size();    
    int i;

    featClust.resize(sz, 0);
    featIdxArr.resize(sz, 0);

    for (i=0; i<sz; i++)
    {
        featIdxArr[i] = i;
    }

    SiftTreeNode nullNode;
    memset(&nullNode, 0, sizeof(SiftTreeNode));
    int nmax = (int)((pow(nsplits, maxLevel+1)-1)/(nsplits-1));
    treeNode.resize(nmax, nullNode);

    KMeansCluster(1, 1, 0, sz);

    return 0;
}


void SiftTree::Quantize(unsigned int *featIdx, const double *sift)
{
    int lvlOff;
    int splitOff;
    int lvl;

    lvlOff   = 1;
    splitOff = 0;
    lvl      = 1;

    while (1)
    {
        double tmpDst;
        double minDst = FeatDist(treeNode[lvlOff+splitOff].center, sift);
        int    minIdx = 0;

        int i;
        for (i=1; i<nsplits; i++)
        {
            int idx = lvlOff + splitOff + i;

            if (treeNode[idx].numChld == 0)
                continue;
            
            tmpDst = FeatDist(treeNode[idx].center, sift);
            if (tmpDst < minDst)
            {
                minDst = tmpDst;
                minIdx = i;
            }
        }

        if (treeNode[lvlOff+splitOff+minIdx].numChld == 1 || lvl == maxLevel)
        {
            *featIdx = lvlOff + splitOff + minIdx;
            break;
        }

        lvl += 1;

        // 数列，呵呵
        lvlOff   = lvlOff * nsplits + 1;
        splitOff = (splitOff + minIdx) * nsplits;
    }
}
