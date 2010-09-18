/* Copyright (c) 2008, Eidgenössische Technische Hochschule Zürich, ETHZ
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the <organization> nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY Eidgenössische Technische Hochschule Zürich, ETHZ "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL the Eidgenössische Technische Hochschule Zürich, ETHZ BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/////////////////////////////////////////////////////////
// Database class: Contains inverted file class and forward file
// Class allows insertation of documents (vw vector) with pixel coordinates into database.
/////////////////////////////////////////////////////////

// Author: Friedrich Fraundorfer, fraundorfer@inf.ethz.ch

#ifndef FF_DATABASE
#define FF_DATABASE

#include "ff_invfile.h"
#include "ff_sort.h"

#define DB_RANSAC 500


class ff_database: public ff_invfile {
public:
	ff_database(void);
	~ff_database(void);

	// Class initialization and memory allocation
	// Param: nrdocs ... (IN) maximum number of documents in database
	//        nrvisualwords ... (IN) range of vw's quantization of voctree
	//        maxvw ....... (IN) max number of vw per document
	int init(int nrdocs, int nrvisualwords, int maxvw);

	// Inserts a document into database. "docname" is a label for the document. On query the label gets returned.
	// Param: vw ... (IN) visual words of the document
	//        nr ... (IN) length of visual word vector
	//        docname ... (IN) label for document
	int insertdoc(unsigned int *vw, int nr, int docname);
	int insertdoc(unsigned int *vw, int nr, int docname, float *x, float *y);

	int match(int docpos, unsigned int *vw, int nr, float *x, float *y, float *x1, float *y1, float *x2, float *y2);
	int getforwarddata(int docpos, float *x, float *y, float *s);
	void normalize(void);

	void clean(void);

private:
	int m_init;
	unsigned int *m_vw; // store visual words index of a document
	int *m_nrvw; // number of visual words in a document
	float *m_x;
	float *m_y;
	float *m_s;
	float *m_X;
	float *m_Y;
	float *m_Z;
	unsigned char *m_sift;
	int *m_blockpt; // start position of vm's of a doc in m_vm
	int m_blockcount;
	ff_heapsort m_vwsorter;
	float **m_svdU;
	float *m_svdW;
	float **m_svdV;
	float **m_Harr;
	float **m_ptarr;
	float *m_gscore;

};

#endif //FF_DATABASE
