#ifndef DATA_DISTRIBUTOR_H_
#define DATA_DISTRIBUTOR_H_

// #include <mpi.h>
#include "boost/mpi.hpp"
#include <zoltan_cpp.h>

#include <algorithm>
#include <vector>
#include <cstdlib>
#include <iostream>

#include "distributed_structures.h"
#include "data_distributor_hypergraph.h"

using namespace std;
namespace mpi = boost::mpi;

static void quit(const char *msg = "") {
	std::cout << msg << std::endl;
	MPI_Finalize();
	exit(0);
}

template<typename L, typename D>
void compute_partitioning_of_hypergraph(std::vector<L> A_csc_row_idx,
		std::vector<L> A_csc_col_ptr, const L m, const L n,
		data_distributor<L, D> &dataDistributor) {

	int argc = 0;
	char **argv = NULL;

//	MPI_Init(&argc, &argv);

	int rank, size;
	//MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// TODO: Check the meaning of MPI_Comm_size
	const L k = size;

	// Call the C function Zoltan_Initialize before using the C++ interface!
	float version;
	int rc = Zoltan_Initialize(argc, argv, &version);
	if (rc != ZOLTAN_OK)
		quit("Zoltan_Initialize failed. Sorry ...\n");
	Zoltan *zz = new Zoltan(MPI_COMM_WORLD);
	if (zz == NULL)
		quit("new Zoltan failed. Sorry ...\n");

	// Pick the partitioning method, e.g.
	// zz->Set_Param( "LB_METHOD", "BLOCK");
	zz->Set_Param("LB_METHOD", "HYPERGRAPH");
	zz->Set_Param("LB_APPROACH", "PARTITION");
	zz->Set_Param("IMBALANCE_TOL", "1.0001");

	// Set some more params
	zz->Set_Param("NUM_GID_ENTRIES", "1"); // 1 integer
	zz->Set_Param("NUM_LID_ENTRIES", "1"); // 1 integer
	zz->Set_Param("OBJ_WEIGHT_DIM", "0"); // no weights

	// Set up the adaptor from CSC to Zoltan's hypergraphs
	printf("BBBBBB %d %d %d\n", rank, A_csc_col_ptr[2], n);
	//  hypergraph<L, D> hg(A_csc_row_idx, A_csc_col_ptr, m, n,rank);

	hypergraph<L, D>* _hg_ptr;
	if (rank == 0) {
		hypergraph<L, D> hg(A_csc_row_idx, A_csc_col_ptr, m, n, rank);
		_hg_ptr = &hg;
	} else {
		A_csc_row_idx.resize(0);
		A_csc_col_ptr.resize(0);
		hypergraph<L, D> hg(A_csc_row_idx, A_csc_col_ptr, 0, 0, rank);
		_hg_ptr = &hg;
	}

	// Set up the call-backs
	printf("AAAAAAAAAAAAAAAAAAA %d \n", rank);

	zz->Set_Num_Obj_Fn(hypergraph<L, D>::get_number_of_vertices,
			(void *) _hg_ptr);
	zz->Set_Obj_List_Fn(hypergraph<L, D>::get_vertex_list, (void *) _hg_ptr);
	zz->Set_HG_Size_CS_Fn(hypergraph<L, D>::get_hypergraph_size,
			(void *) _hg_ptr);
	zz->Set_HG_CS_Fn(hypergraph<L, D>::get_hypergraph, (void *) _hg_ptr);

	//	zz->Set_Num_Obj_Fn(hypergraph<L, D>::get_number_of_vertices, (void *) &hg);
	//	zz->Set_Obj_List_Fn(hypergraph<L, D>::get_vertex_list, (void *) &hg);
	//	zz->Set_HG_Size_CS_Fn(hypergraph<L, D>::get_hypergraph_size, (void *) &hg);
	//	zz->Set_HG_CS_Fn(hypergraph<L, D>::get_hypergraph, (void *) &hg);

	//  return;
	// Variables which capture the partitioning returned by Zoltan
	int changes; // 1 if partitioning was changed, 0 otherwise
	int numGidEntries; // Number of integers used for a global ID
	int numLidEntries; // Number of integers used for a local ID
	int numImport; // Number of vertices to be sent to me
	ZOLTAN_ID_PTR importGlobalIds; // Global IDs of vertices to be sent to me
	ZOLTAN_ID_PTR importLocalIds; // Local IDs of vertices to be sent to me
	int *importProcs; // Process rank for source of each incoming vertex
	int *importToPart; // New partition for each incoming vertex
	int numExport; // Number of vertices I must send to other processes
	ZOLTAN_ID_PTR exportGlobalIds; // Global IDs of the vertices I must send
	ZOLTAN_ID_PTR exportLocalIds; // Local IDs of the vertices I must send
	int *exportProcs; // Process to which I send each of the vertices
	int *exportToPart; // Partition to which each vertex will belong
	printf("CCCC %d \n", rank);
	rc = zz->LB_Partition(changes, numGidEntries, numLidEntries, numImport,
			importGlobalIds, importLocalIds, importProcs, importToPart,
			numExport, exportGlobalIds, exportLocalIds, exportProcs,
			exportToPart);
	if (rc != ZOLTAN_OK) {
		delete zz;
		printf("Process %d failed.\n", rank);
		quit("Partitioning failed.\n");
	}
	printf("DDDD %d \n", rank);

	printf("%d numImport:%d   export%d\n", rank, numImport, numExport);

	if (rank == 0) {
		dataDistributor.init(n, size);
		int rank_0_verteces = n - numExport;
		dataDistributor.countsPtr[1] = rank_0_verteces;
		for (int i = 0; i < numExport; i++) {
			dataDistributor.countsPtr[exportToPart[i] + 1]++;
//			printf("%d %d vertex[%d] belongs to %d\n", rank, i,
//					exportGlobalIds[i], exportToPart[i]);
		}
		for (int i = 1; i < size; i++) {
			dataDistributor.countsPtr[i + 1] = dataDistributor.countsPtr[i]
					+ dataDistributor.countsPtr[i + 1];
		}
		for (int i = 0; i < numExport; i++) {
			dataDistributor.indexes[dataDistributor.countsPtr[exportToPart[i]]]
					= exportGlobalIds[i];
			dataDistributor.countsPtr[exportToPart[i]]++;
		}
		dataDistributor.countsPtr[0] = rank_0_verteces;
		for (int i = size - 1; i > 0; i--) {
			dataDistributor.countsPtr[i] = dataDistributor.countsPtr[i - 1];
		}
		dataDistributor.countsPtr[0] = 0;

		for (int i = 0; i <= size; i++) {
			printf("PTR[%d]=%d\n", i, dataDistributor.countsPtr[i]);
		}
		//Add RowId for RANK 0
		int assigned_to_rank0 = 0;
		for (int vertex = 0; vertex < n; vertex++) {
			bool not_in_others = true;
			for (int rID = dataDistributor.countsPtr[1]; rID <= dataDistributor.countsPtr[size]; rID++) {
				if (dataDistributor.indexes[rID] == vertex) {
					not_in_others = false;
					break;
				}
			}
			if (not_in_others) {
				dataDistributor.indexes[assigned_to_rank0] = vertex;
				assigned_to_rank0++;
			}
		}
		//		printf("Rank 0 ma %d vrcholov\n", assigned_to_rank0);


		// Get info how dataDistributor looks like
		printf("dataDistributor\n");
		for (int sample = 0; sample < dataDistributor.samples_count; sample++) {
			printf("SAmple %d \n", sample);
			for (int sampl = dataDistributor.countsPtr[sample]; sampl
					< dataDistributor.countsPtr[sample + 1]; sampl++) {
				printf("%d ", dataDistributor.indexes[sampl]);
			}
			printf("\n");
		}

	}
	// Transfer the partitioning to dataDistributor
	// Free Zoltan up
	Zoltan::LB_Free_Part(&importGlobalIds, &importLocalIds, &importProcs,
			&importToPart);
	Zoltan::LB_Free_Part(&exportGlobalIds, &exportLocalIds, &exportProcs,
			&exportToPart);
	delete zz;
	MPI::Finalize();
	// NOTE: No "return", as dataDistributor is passed by reference
}

#endif /* DATA_DISTRIBUTOR_H_ */
