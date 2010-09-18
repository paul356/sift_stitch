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

// Author: Friedrich Fraundorfer, fraundorfer@inf.ethz.ch

#include "ff_database.h"
#include <wchar.h>
#include "iostream"
#include "math.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

using namespace std;


ff_database::ff_database(void) {
	m_init = 0;
}

ff_database::~ff_database(void) {
	if (m_init == 1) {
		clean();
		m_init = 0;
	}
}

int ff_database::init(int nrdocs, int nrvisualwords, int maxvw) {

	ff_invfile::init(nrdocs, nrvisualwords, maxvw);
	m_vw = new unsigned int[nrdocs*maxvw];
	m_nrvw = new int[nrdocs];
	m_x = new float[nrdocs*maxvw];
	m_y = new float[nrdocs*maxvw];
	m_blockpt = new int[nrdocs];
	m_blockcount = 0;
	m_svdU = new float*[9];
	for (int i = 0; i < 9; i++)
		m_svdU[i] = new float[9];
	m_svdW = new float[9];
	m_svdV = new float*[9];
	for (int i = 0; i < 9; i++)
		m_svdV[i] = new float[9];

	m_Harr = new float*[DB_RANSAC];
	for (int i = 0; i < DB_RANSAC; i++)
		m_Harr[i] = new float[9];
	m_ptarr = new float*[DB_RANSAC];
	for (int i = 0; i < DB_RANSAC; i++)
		m_ptarr[i] = new float[16];
	m_gscore = new float[DB_RANSAC];

	return 0;
}

void ff_database::clean(void) {
	if (m_init == 1) {
		ff_invfile::clean();
		delete [] m_vw;
		delete [] m_nrvw;
		delete [] m_x;
		delete [] m_y;
		delete [] m_blockpt;
		for (int i = 0; i < 9; i++) {
			delete [] m_svdU[i];
			delete [] m_svdV[i];
		}
		delete [] m_svdU;
		delete [] m_svdW;
		delete [] m_svdV;
		for (int i = 0; i < DB_RANSAC; i++) {
			delete [] m_Harr[i];
			delete [] m_ptarr[i];
		}
		delete [] m_Harr;
		delete [] m_ptarr;
		delete [] m_gscore;

		m_init = 0;
	}
}


int ff_database::insertdoc(unsigned int *vw, int nr, int docname) {
	if (m_docnr == m_maxdocs) return -1;
	// add to forward file
	m_nrvw[m_docnr] = nr;
	m_blockpt[m_docnr] = m_blockcount;
	memcpy(&m_vw[m_blockcount], vw, sizeof(unsigned int)*nr);
	m_blockcount += nr;

	return ff_invfile::insert(vw, nr, docname);
}


int ff_database::insertdoc(unsigned int *vw, int nr, int docname, float *x, float *y) {
	if (m_docnr == m_maxdocs) return -1;
	// add to forward file
	m_nrvw[m_docnr] = nr;
	m_blockpt[m_docnr] = m_blockcount;
	memcpy(&m_vw[m_blockcount], vw, sizeof(unsigned int)*nr);
	memcpy(&m_x[m_blockcount], x, sizeof(float)*nr);
	memcpy(&m_y[m_blockcount], y, sizeof(float)*nr);
	m_blockcount += nr;

	return ff_invfile::insert(vw, nr, docname);
}


int ff_database::match(int docpos, unsigned int *vw, int nr, float *x, float *y, float *x1, float *y1, float *x2, float *y2) {
	// database is considered to be left image
	// database points are returned in x1,y1
	// query points are copied to x2,y2

	int ivwdb;
	int ivw;
	int matchnr = 0;

	ivw = 0;
	ivwdb = m_blockpt[docpos];

	while (ivw < nr && ivwdb-m_blockpt[docpos] < m_nrvw[docpos]) {
		while (vw[ivw] < m_vw[ivwdb]) {
			ivw++;
		}
		while (vw[ivw] > m_vw[ivwdb]) {
			ivwdb++;
		}
		if (vw[ivw] == m_vw[ivwdb]) {   // matching word found; sorted, unique vw list required!
			x1[matchnr] = m_x[ivwdb];
			y1[matchnr] = m_y[ivwdb];
			x2[matchnr] = x[ivw];
			y2[matchnr] = y[ivw];
			matchnr++;
			ivw++;
			ivwdb++;
		}
	}


	return matchnr;
}




int ff_database::getforwarddata(int docpos, float *x, float *y, float *s) {
	int nr = m_nrvw[docpos];
	int m_blockcount = m_blockpt[docpos];
	memcpy(x, &m_x[m_blockcount], sizeof(float)*nr);
	memcpy(y, &m_y[m_blockcount], sizeof(float)*nr);
	memcpy(s, &m_s[m_blockcount], sizeof(float)*nr);
	return nr;
}

void ff_database::normalize(void) {

	for (int j = 0; j < m_docnr; j++) {
		int ivwdb = m_blockpt[j];
		int nr = m_nrvw[j];

		m_docsums[j] = 0;
		m_docnormalizer[j] = 0;
		for (int i = 0; i < nr; i++) {
			m_docindex[m_vw[ivwdb+i]]++;
		}
		for (int i = 0; i < nr; i++) {
			if (m_docindex[m_vw[ivwdb+i]] > 0) {
				m_docsums[j] += ((float)m_docindex[m_vw[ivwdb+i]]*m_idf[m_vw[ivwdb+i]]*(float)m_docindex[m_vw[ivwdb+i]]*m_idf[m_vw[ivwdb+i]]);
				m_docindex[m_vw[ivwdb+i]] = 0;
			}
		}
		m_docnormalizer[j] = (float)1./sqrt(m_docsums[j]);
	}
}
