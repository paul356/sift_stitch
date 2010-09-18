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
// Inverted file class/database structure
// Class allows insertation of documents (vw vector) into inverted file database as well
// of document query. Scoring is weigthed by IDF weights.
/////////////////////////////////////////////////////////

// Author: Friedrich Fraundorfer, fraundorfer@inf.ethz.ch


/*
	//Sample code-snipet:

	// load voctree
	ff_voctree voctree;
	voctree.init("voc1000000dog.bin");

	// quantize SIFT vectors into visual words (vw's)
	for (int i = 0; i < nr; i++) {
		...
		voctree.quantize(&vw, &sift);  // quantizes a single SIFT vector into 1 visual word
		...
	}

	// create inverted file
	ff_invfile inv;
	inv.init(nr, voctree.nrvisualwords(), 2000);

	// add images to database
	for (int i = 0; i < nr; i++) {
		...
		inv.insert(&vw, vwnr, label);   // adds a document vector (consisting of vwnr visual words) to database under the denoted label
		...
	}

	// compute idf weighting for database
	inv.computeidf();

	// query image
	inv.querytopn(&vw, vwnr, 4, qdocnames, qscores);   // returns the label of the 4 closest matches in the database

*/


#ifndef FF_INVFILE
#define FF_INVFILE

#define INV_BLOCKSIZE 10


/*#define __max(x, y) (((x) > (y))?(x):(y))*/
/*#define __min(x, y) (((x) > (y))?(y):(x))*/

inline int __max(int x, int y) 
{
    if (x < y)
        return y;
    else
        return x;
}
inline int __min(int x, int y)
{
    if (x < y)
        return x;
    else
        return y;
}


class ff_invfile {
////
	struct invblock {
		invblock *next;
		unsigned int data[INV_BLOCKSIZE];
	};
public:
	ff_invfile(void);
	virtual ~ff_invfile(void);

	// Class initialization and memory allocation
	// Param: nrdocs ... (IN) maximum number of documents in database
	//        nrvisualwords ... (IN) range of vw's quantization of voctree
	//        maxvw ....... (IN) max number of vw per document
	int init(int nrdocs, int nrvisualwords, int maxvw);

	// Inserts a document into database. "docname" is a label for the document. On query the label gets returned.
	// Param: vw ... (IN) visual words of the document
	//        nr ... (IN) length of visual word vector
	//        docname ... (IN) label for document
	int insert(unsigned int *vw, int nr, int docname);

	// Querying the database. Finds the closest database entry to the given document and returns its label
	// Param: vw ... (IN) visual words of the query document
	//        nr ... (IN) length of visual word vector
	int query(unsigned int *vw, int nr);

	// Querying the database. Finds the n-closest database entries to the given document and returns their labels and scores
	// A vector for holding the labels and scores has to be provided
	// Param: vw ... (IN) visual words of the query document
	//        nr ... (IN) length of visual word vector
	//        sort ..(IN) specifies if docnames are sorted according to scores (sort == 1 ... sorted)
	int querytopn(unsigned int *vw, int nr, int n, int *docnames, float *scores, int sort = 1);


	// Computes IDF weights from current database.
	void computeidf(void);

	// Stores IDF weights to file.
	void saveidf(char *fname);
	// Loads IDF weights from file.
	void loadidf(char *fname);

	void clean(void);

	inline int isvalid(void){return m_init;}

	unsigned int *m_invnr;
protected:
	int m_init;
	invblock* m_blockspace;
	invblock** m_inv;
	int m_nrvisualwords;
	int m_maxdocs;
	int m_maxvw;
	int m_blocknr;
	int m_docnr;
	int m_maxblocks;
	float* m_idf;
	float *m_score;
	float *m_scoreadd;
	int *m_scorepos;
	int *m_docnames;
	float *m_docsums;
	float *m_docnormalizer;
	int *m_docindex;

};

#endif //FF_INVFILE
