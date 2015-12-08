#ifndef ACDC_DISTRIBUTED_COMMON
#define ACDC_DISTRIBUTED_COMMON

#include "distributed_include.h"


#include "distributed_essentials.h"
#include "distributed_structures.h"
#include "data_distributor.h"



template<typename D, typename L>
void gather_x(mpi::communicator world, ProblemData<L, D> &part, ProblemData<L, D> &inst,
		data_distributor<L, D> &dataDistributor) {
	if (world.rank() == 0) {
		std::vector<D> empty_v(0, 0);
		std::vector < std::vector<D> > all_x(world.size(), empty_v);
		gather(world, part.x, all_x, 0);
		inst.x.resize(inst.n, 0);
		for (int i = 0; i < world.size(); i++) {
			int tmp = 0;
			for (int j = dataDistributor.countsPtr[i]; j < dataDistributor.countsPtr[i + 1]; j++) {
				int vertex = dataDistributor.indexes[j];
				inst.x[vertex] = all_x[i][tmp];
				tmp++;
			}
		}
		//				for (int i = 0; i < inst.n; i++) {
		//					printf("x[%i]=%f ", i, inst.x[i]);
		//				}
	} else {
		gather(world, part.x, 0);
	}
}

template<typename T, typename I>
void getCSC_from_COO_plus_find_size(const std::vector<T> &h_Z_values, const std::vector<I> &h_Z_row_idx,
		const std::vector<I>& h_Z_col_idx, std::vector<T>& Z_csc_val, std::vector<I>& Z_csc_rowIdx,
		std::vector<I>& Z_csc_ColPtr, int &m, int &n, data_distributor<I, T> &dataDistributor, I partId) {
	unsigned long long nnz = h_Z_values.size();
	Z_csc_val.resize(nnz);
	Z_csc_rowIdx.resize(nnz);
	m = -1;
	n = 0;

	for (I i = 0; i < dataDistributor.columns_parts.size(); i++) {
		if (dataDistributor.columns_parts[i] == partId)
			n++;
	}

	std::vector<I> local_mapping = dataDistributor.columns_parts;
	n = 0;
	for (I i = 0; i < dataDistributor.columns_parts.size(); i++) {
		if (dataDistributor.columns_parts[i] == partId) {
			local_mapping[i] = n;
			n++;
		}
	}
	for (I i = 0; i < nnz; i++) {
		if (h_Z_row_idx[i] > m)
			m = h_Z_row_idx[i];
	}
	m++;
	Z_csc_ColPtr.resize(n + 1);
	for (int i = 0; i < nnz; i++) {
		Z_csc_ColPtr[local_mapping[h_Z_col_idx[i]]]++;
	}
	//	// Get the same for csc
	for (int i = 0; i < n; i++) {
		Z_csc_ColPtr[i + 1] += Z_csc_ColPtr[i];
	}
	for (int i = n; i > 0; i--) {
		Z_csc_ColPtr[i] = Z_csc_ColPtr[i - 1];
	}
	Z_csc_ColPtr[0] = 0;
	// ========Fill Data into correct format
	for (int i = 0; i < nnz; i++) {
		Z_csc_val[Z_csc_ColPtr[local_mapping[h_Z_col_idx[i]]]] = h_Z_values[i];
		Z_csc_rowIdx[Z_csc_ColPtr[local_mapping[h_Z_col_idx[i]]]] = h_Z_row_idx[i];
		Z_csc_ColPtr[local_mapping[h_Z_col_idx[i]]]++;
	}
	for (int i = n; i > 0; i--) {
		Z_csc_ColPtr[i] = Z_csc_ColPtr[i - 1];
	}
	Z_csc_ColPtr[0] = 0;
}

template<typename D, typename L>
void load_data(ProblemData<L, D> &inst, mpi::communicator &world, DistributedSettings &settings) {
	inst.lambda = 1;
	D omega = 5; //FIXME compute TRUE eigenvalue or get omega upperbound!
	inst.sigma = 1 + (omega - 1) * (settings.iterationsPerThread * settings.totalThreads * world.size())
			/ inst.n;

	if (world.rank() == 0) {
		int m;
		int n;
#define READ_SVN
#ifdef READ_SVN

#ifdef ONEDDIE
		const char *filename = "/exports/work/scratch/taki/rcv1_train";
#else
		//		const char *filename = "/home/jmarecek/projects/ac-dc/DATA/svm/rcv1_industries_train.svm";
		const char* filename = "/home/taki/projects/ac-dc/DATA/SVM/rcv1_train.binary";
#endif
		//			const char* filename = "/home/taki/projects/resources/atest.dat";
		//		const char* filename = "/tmp/asfasdfa";
		//	const char* filename = "/document/test.dat";
		int nclasses = 2;
		long long nnz = 0;
		parse_LIB_SVM_data_get_size(filename, m, n, nnz);
		// Load training data
		std::cout << "Loading SVM with m = " << m << ", n = " << n << ", nnz = " << nnz << std::endl;
		int status = parseLibSVMdata(filename, inst.A_csc_values, inst.A_csc_row_idx, inst.A_csc_col_ptr, inst.b,
				m, n, nclasses, nnz);
		inst.n = n;
		inst.m = m;

		//		std::vector<L> buffer = inst.A_csc_col_ptr;
		//	L j = 0;
		//	for (L i = 0; i < n; i++) {
		//		if (buffer[i + 1] - buffer[i] > 0) {
		//			inst.A_csc_col_ptr[j] = buffer[i];
		//			j++;
		//		} else {
		//		}
		//	}
		//	inst.A_csc_col_ptr[j] = buffer[n];
		//	inst.A_csc_col_ptr.resize(j + 1);
		//	inst.n = j + 1;
		n = inst.n;

		//	//Print CSC data
		//	for (L i = 0; i < n; i++) {
		//		printf("column %d: ", i);
		//		for (L j = inst.A_csc_col_ptr[i]; j < inst.A_csc_col_ptr[i + 1]; j++) {
		//			printf("%d ", inst.A_csc_row_idx[j]);
		//		}
		//		printf("\n");
		//	}

#else
		//	  generate_k_diagonal(inst, m, n, 10, 3);
		// nesterov_generator(inst);
		//	 generate_k_diagonal(inst, m, n, 100, 5);

		//	generate_k_diagonal_with_few_full_columns(inst, m, n, 100, 3);
		//		printf("Loading a tridiagonal problem with m = %d, n = %d \n", inst.m, inst.n);
		//		m = 100; n = 100;
		//		generate_k_diagonal(inst, m, n, 100, 5);
		//		inst.n = n;
		//		inst.m = m;

		generate_random_problem(inst);
#endif
	} else {
		inst.m = 0;
		inst.n = 0;
	}
}

template<typename D, typename L>
void generate_data_with_know_optimal_value(ProblemData<L, D> &instOut, mpi::communicator &world,
		DistributedSettings &settings) {
	ProblemData<L, D> inst;
	if (world.rank() == 0) {
		generate_random_known(inst);
	} else {
		inst.n = 0;
		inst.m = 0;
	}
	instOut.lambda = 1;

	instOut.n = inst.n;
	instOut.m = inst.m;
	instOut.b = inst.b;

	getCSR_from_CSC(inst.A_csc_values, inst.A_csc_row_idx, inst.A_csc_col_ptr, instOut.A_csr_values,
			instOut.A_csr_col_idx, instOut.A_csr_row_ptr, inst.m, inst.n);

}

template<typename D, typename L>
void generate_data(ProblemData<L, D> &inst, mpi::communicator &world, generator_data& gen) {
	if (world.rank() == 0) {
		if (gen.type == Blocked)
			gen.k = world.size();
		generate_instance(inst, gen);
	} else {
		inst.n = 0;
		inst.m = 0;
	}
}

template<typename D, typename L>
void load_data_from_multiple_sources(ProblemData<L, D> &inst, mpi::communicator &world,
		DistributedSettings &settings, const char* filename) {
	inst.lambda = 1;
	L m;
	L n;
	char new_filename[255];
	n = sprintf(new_filename, "%s%d", filename, world.rank());
	int nclasses = 2;
	long long nnz = 0;

	world.barrier();
	printf("I am node %d and i am going to get the file size info free: %d\n",world.rank(),getVirtualMemoryCurrentlyUsedByCurrentProcess());

	world.barrier();

	parse_LIB_SVM_data_get_size(new_filename, m, n, nnz);
	// Load training data
	printf("NODE %d loaded SVM with m = %d, n = %d, nnz = %d \n", world.rank(), m, n, nnz);
	printf("I am node %d and i am done with file size info free: %d\n",world.rank(),getVirtualMemoryCurrentlyUsedByCurrentProcess());

	world.barrier();


	int status = parse_lib_SVM_data_into_CSR(new_filename, inst.A_csr_values, inst.A_csr_col_idx,
			inst.A_csr_row_ptr, inst.b, m, n, nclasses, nnz,world);

	printf("I am node %d and i parsed free: %d\n",world.rank(),getVirtualMemoryCurrentlyUsedByCurrentProcess());

	world.barrier();



	inst.n = n;
	inst.m = m;
	D omega = 5; //FIXME compute TRUE eigenvalue or get omega upperbound!
	inst.sigma = 1 + (omega - 1) * (settings.iterationsPerThread * settings.totalThreads * world.size())
			/ inst.n;
}

template<typename D, typename L>
void load_data_from_multiple_sources(ProblemData<L, D> &inst, mpi::communicator &world,
		DistributedSettings &settings) {
	inst.lambda = .0001;
	L m;
	L n;
	//	const char* filename = "/home/taki/projects/ac-dc/DATA/SVM/a5a/a5a";
	//	const char* filename = "/home/taki/projects/resources/atest.dat";
	const char* filename = "/home/taki/projects/ac-dc/DATA/SVM/rcv1_train.binary";
	char new_filename[255];
	n = sprintf(new_filename, "%s%d", filename, world.rank());
	int nclasses = 2;
	long long nnz = 0;
	parse_LIB_SVM_data_get_size(new_filename, m, n, nnz);
	// Load training data
	printf("NODE %d loaded SVM with m = %d, n = %d, nnz = %d \n", world.rank(), m, n, nnz);
	int status = parse_lib_SVM_data_into_CSR(new_filename, inst.A_csr_values, inst.A_csr_col_idx,
			inst.A_csr_row_ptr, inst.b, m, n, nclasses, nnz);
	inst.n = n;
	inst.m = m;
	//	//Print CSR data
	for (L i = 0; i < m; i++) {
		//			printf("row %d: ", i);
		//			for (L j = inst.A_csr_row_ptr[i]; j < inst.A_csr_row_ptr[i + 1]; j++) {
		//				printf("%d ", inst.A_csr_col_idx[j]);
		//			}
		//			printf("\n");
	}
	D omega = 5; //FIXME compute TRUE eigenvalue or get omega upperbound!
	inst.sigma = 1 + (omega - 1) * (settings.iterationsPerThread * settings.totalThreads * world.size())
			/ inst.n;
}
void configure_Zoltan(Zoltan *zz, DistributedSettings &settings) {

	// We want to start from scratch, as all data are at the root node
	zz->Set_Param("LB_APPROACH", "PARTITION");

	// Pick the partitioning method, e.g.
	zz->Set_Param("LB_METHOD", "HYPERGRAPH");

	//	 zz->Set_Param("LB_METHOD", "BLOCK");  // WOULD NEED OTHER CALLBACKS

	zz->Set_Param("HYPERGRAPH_PACKAGE", "PHG");/* version of method */
	// 	zz->Set_Param("PHG_CUT_OBJECTIVE", "HYPEREDGES");/* version of method */
	// 	zz->Set_Param("PHG_OUTPUT_LEVEL", "1");/* version of method */
	//		zz->Set_Param("CHECK_HYPERGRAPH", "1");/* version of method */

	//	zz->Set_Param("RETURN_LISTS", "ALL");/* export AND import lists */
	// Set some more params
	zz->Set_Param("IMBALANCE_TOL", "1.50");
	zz->Set_Param("NUM_GID_ENTRIES", "1"); // 1 integer
	zz->Set_Param("NUM_LID_ENTRIES", "1"); // 1 integer
	zz->Set_Param("OBJ_WEIGHT_DIM", "0"); // no weights

	if (settings.verbose) {
		zz->Set_Param("DEBUG_LEVEL", "2");
	} else {
		zz->Set_Param("DEBUG_LEVEL", "0");
	}
	//	zz->Set_Param("DEBUG_LEVEL", "5");
}

template<typename D, typename L>
void distribute_data(mpi::communicator &world, data_distributor<L, D> &dataDistributor,
		ProblemData<L, D> &inst, ProblemData<L, D> &part, DistributedSettings &settings) {
	world.barrier();
	Zoltan *zz = new Zoltan(world); // MPI_COMM_WORLD
	if (zz == NULL)
		quit("new Zoltan failed. Sorry ...\n");
	configure_Zoltan(zz, settings);
	// Set up the adaptor from CSC to Zoltan's hypergraphs
	hypergraph<L, D>* hg;
	if (world.rank() == 0) {
		hg = new hypergraph<L, D> (inst.A_csc_row_idx, inst.A_csc_col_ptr, inst.m, inst.n, world.rank());
	} else {
		std::vector<L> empty_vector;
		hg = new hypergraph<L, D> (empty_vector, empty_vector, 0, 0, world.rank());
	}
	if (settings.verbose)
		printf("Zoltan callback setting in thread %d \n", world.rank());
	// Set up the call-backs
	zz->Set_Num_Obj_Fn(hypergraph<L, D>::get_number_of_vertices, (void *) hg);
	zz->Set_Obj_List_Fn(hypergraph<L, D>::get_vertex_list, (void *) hg);
	zz->Set_HG_Size_CS_Fn(hypergraph<L, D>::get_hypergraph_size, (void *) hg);
	zz->Set_HG_CS_Fn(hypergraph<L, D>::get_hypergraph, (void *) hg);
	if (settings.verbose)
		printf("Zoltan partitioning in thread %d \n", world.rank());
	// Variables which capture the output of the partitioning *returned* by Zoltan
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
	int rc = zz->LB_Partition(changes, numGidEntries, numLidEntries, numImport, importGlobalIds,
			importLocalIds, importProcs, importToPart, numExport, exportGlobalIds, exportLocalIds, exportProcs,
			exportToPart);
	if (rc != ZOLTAN_OK) {
		delete zz;
		printf("Partitioning failed in process %d.\n", world.rank());
		quit("Partitioning failed.\n");
	}
	// In the following, we create the "sub-instance" to be solved at *at each node*
	// This consists of a vector describing the columns chosen for the "sub-instance":
	std::vector<L> part_col_idx;
	// and then the same data structure as for the complete instance:

	// where we fill in first:
	part.b = inst.b;
	part.m = inst.m;

	//	printf("===%d I have to import %d  export %d\n", rank, numImport, numExport);

#ifndef THE_PERFECT_COMPILER
	const int reqs_each = 6;
#else // if THE_PERFECT_COMPILER
	const int reqs_each = 4;
#endif

	if (world.rank() == 0) {
		dataDistributor.init(inst.n, world.size());
		dataDistributor.countsPtr[1] = inst.n - numExport;
		for (int i = 0; i < numExport; i++) {
			dataDistributor.countsPtr[exportToPart[i] + 1]++;
			// printf("%d %d vertex[%d] belongs to %d\n", rank, i, exportGlobalIds[i], exportToPart[i]);
		}
		for (int i = 1; i < world.size(); i++) {
			dataDistributor.countsPtr[i + 1] = dataDistributor.countsPtr[i] + dataDistributor.countsPtr[i + 1];
		}
		for (int i = 0; i < numExport; i++) {
			dataDistributor.indexes[dataDistributor.countsPtr[exportToPart[i]]] = exportGlobalIds[i];
			dataDistributor.countsPtr[exportToPart[i]]++;
		}
		dataDistributor.countsPtr[0] = inst.n - numExport;
		for (int i = world.size() - 1; i > 0; i--) {
			dataDistributor.countsPtr[i] = dataDistributor.countsPtr[i - 1];
		}
		dataDistributor.countsPtr[0] = 0;
		//#ifdef PRINT_PTR_DATA
		//				for (int i = 0; i <= world.size(); i++) {
		//					printf("PTR[%d]=%d\n", i, dataDistributor.countsPtr[i]);
		//				}
		//#endif
		//#ifdef PRINT_PARTITION_BEFORE_SEND
		//		for (int part = 0; part < dataDistributor.samples_count; part++) {
		//			printf("Part %d: ", part);
		//			for (int p = dataDistributor.countsPtr[part]; p < dataDistributor.countsPtr[part + 1]; p++) {
		//				printf("%d ", dataDistributor.indexes[p]);
		//			}
		//			printf("\n");
		//		}
		//#endif
		if (settings.verbose)
			printf("The root node is ready to schedule the transfers for scattering the data.\n");
		mpi::request reqs[reqs_each * world.size()];
		int r = 0;
		// Send the data corresponding to the partition created by Zoltan
		for (int send_to = 1; send_to < world.size(); send_to++) {

			int n_part = dataDistributor.countsPtr[send_to + 1] - dataDistributor.countsPtr[send_to];
			int nnz_part = 0;
			for (int i = dataDistributor.countsPtr[send_to]; i < dataDistributor.countsPtr[send_to + 1]; i++) {
				int j = dataDistributor.indexes[i];
				nnz_part += inst.A_csc_col_ptr[j + 1] - inst.A_csc_col_ptr[j];
			}

			if (settings.verbose)
				printf("Scheduling transfer of %d elements (dimension %d) from %d to %d\n", nnz_part, n_part,
						world.rank(), send_to);

			std::vector<L> col_idx_to_send(n_part);
			std::vector<L> col_ptr_to_send(n_part + 1);
			std::vector<L> row_ind_to_send(nnz_part);
			std::vector<D> values_to_send(nnz_part);

			int v_cnt = 0; // id of an vertex, local within a part
			int nnz_cnt = 0; // id of an nnz, local within a part

			for (int i = dataDistributor.countsPtr[send_to]; i < dataDistributor.countsPtr[send_to + 1]; i++) {

				int j = dataDistributor.indexes[i];

				col_idx_to_send[v_cnt] = j;
				col_ptr_to_send[v_cnt] = nnz_cnt;

				for (int nnz_i = inst.A_csc_col_ptr[j]; nnz_i < inst.A_csc_col_ptr[j + 1]; nnz_i++) {
					row_ind_to_send[nnz_cnt] = inst.A_csc_row_idx[nnz_i];
					values_to_send[nnz_cnt] = inst.A_csc_values[nnz_i];
					nnz_cnt += 1;
				}
				v_cnt += 1;
			}

			// this is [n_part], which is fine as the size is (n_part + 1);
			col_ptr_to_send[n_part] = nnz_part;

			// NOTE: gcc 4.6.1 fails to compile the code if boost tries to resize the vectors automagically
			int tag = (send_to - 1) * reqs_each;
#ifndef THE_PERFECT_COMPILER
			world.send(send_to, tag++, n_part);
			world.send(send_to, tag++, nnz_part);
#endif
			reqs[r++] = world.isend(send_to, tag++, col_idx_to_send);
			reqs[r++] = world.isend(send_to, tag++, col_ptr_to_send);
			reqs[r++] = world.isend(send_to, tag++, row_ind_to_send);
			reqs[r++] = world.isend(send_to, tag++, values_to_send);
			broadcast(world, inst.m, 0);
			broadcast(world, inst.lambda, 0);
			broadcast(world, inst.sigma, 0);
		}
		if (settings.verbose)
			std::cout << "Processing the data that should stay at the root node. " << endl;
		// NOTE: This is not terribly efficient, but we would be waiting anyway
		{
			//Add RowId for RANK 0
			part_col_idx.resize(inst.n - numExport);
			int nnz_cnt = 0; // id of an nnz, local within a part
			int assigned_to_rank0 = 0;
			for (int vertex = 0; vertex < inst.n; vertex++) {
				//				printf("Considering %d  %d %d \n",vertex,dataDistributor.countsPtr[1],dataDistributor.countsPtr[size]);
				bool not_in_others = true;
				for (int rID = dataDistributor.countsPtr[1]; rID < dataDistributor.countsPtr[world.size()]; rID++) {
					if (dataDistributor.indexes[rID] == vertex) {
						not_in_others = false;
						break;
					}
				}
				if (not_in_others) {
					//					printf("assign %d vertex %d\n", assigned_to_rank0, vertex);
					dataDistributor.indexes[assigned_to_rank0] = vertex;
					part_col_idx[assigned_to_rank0] = vertex;
					assigned_to_rank0++;
					nnz_cnt += inst.A_csc_col_ptr[vertex + 1] - inst.A_csc_col_ptr[vertex];
				}
			}
			part.n = assigned_to_rank0;
			part.A_csc_col_ptr.resize(part.n + 1, 0);
			int v_cnt = 0; // id of an vertex, local within a part
			part.A_csc_row_idx.resize(nnz_cnt);
			part.A_csc_values.resize(nnz_cnt);
			nnz_cnt = 0;
			for (int i = 0; i < part.n; i++) {
				int j = part_col_idx[i];
				part_col_idx[i] = j;
				part.A_csc_col_ptr[i] = nnz_cnt;
				for (int nnz_i = inst.A_csc_col_ptr[j]; nnz_i < inst.A_csc_col_ptr[j + 1]; nnz_i++) {
					part.A_csc_row_idx[nnz_cnt] = inst.A_csc_row_idx[nnz_i];
					part.A_csc_values[nnz_cnt] = inst.A_csc_values[nnz_i];
					nnz_cnt++;
				}
			}
			part.A_csc_col_ptr[part.n] = nnz_cnt;
		}
		mpi::wait_all(reqs, reqs + r);
		if (settings.verbose)
			std::cout << "Process " << world.rank() << " has finished sending data. " << endl;
	} else {
		// Nodes other than the root node are receiving data
		if (settings.verbose)
			std::cout << "Process " << world.rank() << " starts receiving data. " << endl;
		mpi::request reqs[reqs_each];
		int tag = (world.rank() - 1) * reqs_each;
		int r = 0;

		// NOTE: gcc 4.6.1 fails to compile the code if boost tries to resize the data automagically
#ifndef THE_PERFECT_COMPILER
		int n_part;
		world.recv(0, tag++, n_part);
		part_col_idx.resize(n_part);
		part.A_csc_col_ptr.resize(n_part + 1);
		int nnz_part;
		world.recv(0, tag++, nnz_part);
		part.A_csc_row_idx.resize(nnz_part);
		part.A_csc_values.resize(nnz_part);
#endif
		reqs[r++] = world.irecv(0, tag++, part_col_idx);
		reqs[r++] = world.irecv(0, tag++, part.A_csc_col_ptr);
		reqs[r++] = world.irecv(0, tag++, part.A_csc_row_idx);
		reqs[r++] = world.irecv(0, tag++, part.A_csc_values);
		broadcast(world, part.m, 0);
		broadcast(world, part.lambda, 0);
		broadcast(world, part.sigma, 0);
		mpi::wait_all(reqs, reqs + r);
		if (settings.verbose)
			std::cout << "Process " << world.rank() << " has finished receiving data. " << endl;
	}
	// TODO: Shall we use part_col_idx to set the dimension and resize x appropriately
	part.n = part_col_idx.size();
	part.x.resize(part.n, 0);
	// NOTE: Distribute b
#ifndef THE_PERFECT_COMPILER
	if (world.rank() == 0) {
		part.m = inst.m;
	}
	broadcast(world, part.m, 0);
	part.b.resize(part.m);
#endif
	broadcast(world, part.b, 0);

	if (settings.verbose)
		printf("Process %d has reached the pre-solving barrier.\n", world.rank());
	world.barrier();

	// TODO: We want to free Zoltan-allocated memory before we allocate solver-specific memory
	if (settings.verbose)
		printf("Freeing Zoltan in process %d.\n", world.rank());
	Zoltan::LB_Free_Part(&importGlobalIds, &importLocalIds, &importProcs, &importToPart);
	Zoltan::LB_Free_Part(&exportGlobalIds, &exportLocalIds, &exportProcs, &exportToPart);
	delete hg;
	world.barrier();
	delete zz;
}

template<typename D, typename L>
void compute_distribution_partitions(mpi::communicator &world, data_distributor<L, D> &dataDistributor,
		ProblemData<L, D> &some_rows, DistributedSettings &settings, distributed_statistics &stat,
		std::vector<L> &ms) {
	int numExport; // Number of vertices I must send to other processes
	ZOLTAN_ID_PTR exportGlobalIds; // Global IDs of the vertices I must send
	ZOLTAN_ID_PTR importGlobalIds; // Global IDs of vertices to be sent to me
	ZOLTAN_ID_PTR importLocalIds; // Local IDs of vertices to be sent to me
	ZOLTAN_ID_PTR exportLocalIds; // Local IDs of the vertices I must send

	int *exportToPart; // Partition to which each vertex will belong
	int *importProcs; // Process rank for source of each incoming vertex
	int *importToPart; // New partition for each incoming vertex
	int *exportProcs; // Process to which I send each of the vertices

	Zoltan *zz = new Zoltan(world); // MPI_COMM_WORLD
	if (zz == NULL)
		quit("new Zoltan failed. Sorry ...\n");
	configure_Zoltan(zz, settings);
	// Call the C function Zoltan_Initialize before using the C++ interface!

	// Set up the adaptor from CSC to Zoltan's hypergraphs
	hypergraph_in_CSR<L, D>* hg;
	//	printf("NODE %d   m= %d  n=%d\n", world.rank(), inst.m, inst.n);


	stat.instance_allocates = stat.instance_nnz * sizeof(int) * 2 /* sparse */* 3 /* Zoltan, us, buffers */;
	if (settings.partitioning == ZoltanPartitioning && stat.instance_allocates
			> settings.allocation_treshold_for_multilevel) {
		std::cout << "Note: Forcing ZoltanMultilevel for instance of size ~ " << stat.instance_allocates << " > "
				<< settings.allocation_treshold_for_multilevel << std::endl;
		settings.partitioning = ZoltanMultilevelPartitioning;
	}
	switch (settings.partitioning) {

	case RandomPartitioning:

		if (world.rank() == 0) {
			for (int i = 0; i < dataDistributor.columns_parts.size(); i++) {
				dataDistributor.columns_parts[i] = (float) world.size() * rand() / (RAND_MAX + 0.01);
			}
		}
		break;

	case BlockedPartitioning:

		if (world.rank() == 0) {
			L total_lenght = dataDistributor.columns_parts.size();
			L total_lenght_per_partition = -1;
			if (total_lenght % world.size() == 0) //FIXME check if this makes sense
				total_lenght_per_partition = total_lenght / world.size();
			else
				total_lenght_per_partition = total_lenght / world.size() + 1;

			for (int i = 0; i < total_lenght; i++) {
				dataDistributor.columns_parts[i] = i / total_lenght_per_partition;
			}
		}
		break;

	case ZoltanMultilevelPartitioning:
		zz->Set_Param("PHG_MULTILEVEL", "1");
		// no break here, we go on

	case ZoltanPartitioning:

		world.barrier();
		hg = new hypergraph_in_CSR<L, D> (some_rows.A_csr_col_idx, some_rows.A_csr_row_ptr, some_rows.m,
				some_rows.n, world.rank(), stat.instance_columns, ms, dataDistributor.columns_parts);

		if (settings.verbose)
			printf("Zoltan callback setting in thread %d \n", world.rank());
		// Set up the call-backs
		zz->Set_Num_Obj_Fn(hypergraph_in_CSR<L, D>::get_number_of_vertices, (void *) hg);
		zz->Set_Obj_List_Fn(hypergraph_in_CSR<L, D>::get_vertex_list, (void *) hg);
		zz->Set_HG_Size_CS_Fn(hypergraph_in_CSR<L, D>::get_hypergraph_size, (void *) hg);
		zz->Set_HG_CS_Fn(hypergraph_in_CSR<L, D>::get_hypergraph, (void *) hg);

		if (settings.verbose)
			printf("Zoltan partitioning in thread %d \n", world.rank());
		// Variables which capture the output of the partitioning *returned* by Zoltan
		int changes; // 1 if partitioning was changed, 0 otherwise
		int numGidEntries; // Number of integers used for a global ID
		int numLidEntries; // Number of integers used for a local ID
		int numImport; // Number of vertices to be sent to me

		world.barrier();
		int rc = zz->LB_Partition(changes, numGidEntries, numLidEntries, numImport, importGlobalIds,
				importLocalIds, importProcs, importToPart, numExport, exportGlobalIds, exportLocalIds, exportProcs,
				exportToPart);
		if (rc != ZOLTAN_OK) {
			delete zz;
			printf("Partitioning failed in process %d.\n", world.rank());
			quit("Partitioning failed.\n");
		}

		if (settings.verbose) {
			printf("Node %d   I have to export %d and import %d vertices \n", world.rank(), numExport, numImport);
		}
		//	printf("total_number_of_columns %d \n", total_number_of_columns);
		if (world.rank() == 0) {
			for (int i = 0; i < numExport; i++) {
				dataDistributor.columns_parts[exportGlobalIds[i]] = exportToPart[i];
			}
			for (int i = 0; i < numImport; i++) {
				dataDistributor.columns_parts[importGlobalIds[i]] = importToPart[i];
			}
		}

		break;
	}

	// TODO: We want to free Zoltan-allocated memory before we allocate solver-specific memory
	if (settings.partitioning >= ZoltanPartitioning) {
		if (settings.verbose)
			printf("Freeing Zoltan in process %d.\n", world.rank());
		Zoltan::LB_Free_Part(&importGlobalIds, &importLocalIds, &importProcs, &importToPart);
		Zoltan::LB_Free_Part(&exportGlobalIds, &exportLocalIds, &exportProcs, &exportToPart);
		delete hg;
		world.barrier();
		delete zz;
	}
	broadcast(world, dataDistributor.columns_parts, 0);
}

template<typename D, typename L>
void obtain_row_shift_data(std::vector<L> &ms, mpi::communicator &world, ProblemData<L, D> &some_rows,
		data_distributor<L, D> &dataDistributor) {
	gather(world, some_rows.m, ms, 0);
	if (world.rank() == 0) {
		std::vector<L> tmp = ms;
		for (L i = 1; i < world.size(); i++) {
			tmp[i] += tmp[i - 1];
		}
		for (L i = 1; i < world.size(); i++) {
			ms[i] = tmp[i - 1];
		}
		ms[0] = 0;
	}
	broadcast(world, ms, 0);
	dataDistributor.global_row_id_mapper = ms;
}

/*
 * In the following, we create the "sub-instance" to be solved at *at each node*
 * This consists of a vector describing the columns chosen for the "sub-instance"
 * and then the same data structure as for the complete instance:
 */
template<typename D, typename L>
void exchange_data_and_fill_my_part_data_in_COO(std::vector<L> &receive_COO_row_id,
		std::vector<L> &receive_COO_col_id, std::vector<D> &receive_COO_vals, std::vector<L> &ms,
		mpi::communicator &world, ProblemData<L, D> &some_rows, data_distributor<L, D> &dataDistributor) {
	const int reqs_each = 6;
	std::vector<L> nnz_elements_which_has_to_be_sent(world.size(), 0);
	std::vector < std::vector<L> > all_nnz_elements_which_has_to_be_sent;
	std::vector<L> point_to_begining_of_data_to_sent(world.size() + 1, 0);
	std::vector<L> point_to_begining_of_data_to_be_received(world.size() + 1, 0);
	// obtain data volumes which has to be exchanged P2P
	for (L col = 0; col < some_rows.A_csr_col_idx.size(); col++) {
		nnz_elements_which_has_to_be_sent[dataDistributor.columns_parts[some_rows.A_csr_col_idx[col]]]++;
	}
	//	printf("RRR %d  %d   %d \n", world.rank(), getPhysicalMemoryCurrentlyUsedByCurrentProcess(),
	//			getVirtualMemoryCurrentlyUsedByCurrentProcess());
	//	world.barrier();
	gather(world, nnz_elements_which_has_to_be_sent, all_nnz_elements_which_has_to_be_sent, 0);
	broadcast(world, all_nnz_elements_which_has_to_be_sent, 0);
	// allocate pointers to data which should be sent and receive
	for (int i = 0; i < world.size(); i++) {
		point_to_begining_of_data_to_be_received[i + 1] = point_to_begining_of_data_to_be_received[i];
		point_to_begining_of_data_to_be_received[i + 1] += all_nnz_elements_which_has_to_be_sent[i][world.rank()];
		point_to_begining_of_data_to_sent[i + 1] = point_to_begining_of_data_to_sent[i];
		if (i != world.rank()) {
			point_to_begining_of_data_to_sent[i + 1] += nnz_elements_which_has_to_be_sent[i];
		}
	}
	receive_COO_row_id.resize(point_to_begining_of_data_to_be_received[world.size()]);
	receive_COO_col_id.resize(point_to_begining_of_data_to_be_received[world.size()]);
	receive_COO_vals.resize(point_to_begining_of_data_to_be_received[world.size()]);
	std::vector<L> sent_COO_row_id(point_to_begining_of_data_to_sent[world.size()]);
	std::vector<L> sent_COO_col_id(point_to_begining_of_data_to_sent[world.size()]);
	std::vector<D> sent_COO_vals(point_to_begining_of_data_to_sent[world.size()]);
	L tmp_my_data = point_to_begining_of_data_to_be_received[world.rank()];
	for (L row = 0; row < some_rows.m; row++) {
		for (L tmp = some_rows.A_csr_row_ptr[row]; tmp < some_rows.A_csr_row_ptr[row + 1]; tmp++) {
			L col = some_rows.A_csr_col_idx[tmp];
			L part = dataDistributor.columns_parts[col];
			if (part != world.rank()) {
				sent_COO_row_id[point_to_begining_of_data_to_sent[part]] = row + ms[world.rank()];
				sent_COO_col_id[point_to_begining_of_data_to_sent[part]] = col;
				sent_COO_vals[point_to_begining_of_data_to_sent[part]] = some_rows.A_csr_values[tmp];
				point_to_begining_of_data_to_sent[part]++;
			} else {
				receive_COO_row_id[tmp_my_data] = row + ms[world.rank()];
				receive_COO_col_id[tmp_my_data] = col;
				receive_COO_vals[tmp_my_data] = some_rows.A_csr_values[tmp];
				tmp_my_data++;
			}
		}
	}
	// decrease pointers to sent (which were incremented in loop above
	for (int i = world.size(); i > 0; i--) {
		point_to_begining_of_data_to_sent[i] = point_to_begining_of_data_to_sent[i - 1];
	}
	point_to_begining_of_data_to_sent[0] = 0;
	//sent and receive data
	mpi::request reqs_exch[6 * (world.size() - 1)];
	int tot_req = 0;
	for (L partTo = 0; partTo < world.size(); partTo++) {
		if (partTo != world.rank()) {
			reqs_exch[tot_req++] = visend(world, partTo, 0,
					&sent_COO_row_id[point_to_begining_of_data_to_sent[partTo]],
					point_to_begining_of_data_to_sent[partTo + 1] - point_to_begining_of_data_to_sent[partTo]);
			reqs_exch[tot_req++] = visend(world, partTo, 1,
					&sent_COO_col_id[point_to_begining_of_data_to_sent[partTo]],
					point_to_begining_of_data_to_sent[partTo + 1] - point_to_begining_of_data_to_sent[partTo]);
			reqs_exch[tot_req++] = visend(world, partTo, 2,
					&sent_COO_vals[point_to_begining_of_data_to_sent[partTo]],
					point_to_begining_of_data_to_sent[partTo + 1] - point_to_begining_of_data_to_sent[partTo]);

			reqs_exch[tot_req++] = virecv(
					world,
					partTo,
					0,
					&receive_COO_row_id[point_to_begining_of_data_to_be_received[partTo]],
					point_to_begining_of_data_to_be_received[partTo + 1]
							- point_to_begining_of_data_to_be_received[partTo]);
			reqs_exch[tot_req++] = virecv(
					world,
					partTo,
					1,
					&receive_COO_col_id[point_to_begining_of_data_to_be_received[partTo]],
					point_to_begining_of_data_to_be_received[partTo + 1]
							- point_to_begining_of_data_to_be_received[partTo]);
			reqs_exch[tot_req++] = virecv(
					world,
					partTo,
					2,
					&receive_COO_vals[point_to_begining_of_data_to_be_received[partTo]],
					point_to_begining_of_data_to_be_received[partTo + 1]
							- point_to_begining_of_data_to_be_received[partTo]);
		}
	}
	mpi::wait_all(reqs_exch, reqs_exch + tot_req);
}

template<typename D, typename L>
void load_distributed_partitions(mpi::communicator &world, data_distributor<L, D> &dataDistributor,
		ProblemData<L, D> &some_columns, DistributedSettings &settings, distributed_statistics &stat,
		string final_file_prefix) {

	FILE* inputfiles[world.size()];
	std::vector<int> nnzInFiles(world.size(), 0);
	unsigned long total_nnz = 0;
	for (int i = 0; i < world.size(); i++) {
		std::stringstream filenameToReadFrom;
		filenameToReadFrom << final_file_prefix << world.rank() << "_" << i << ".dat";
		string tmp = filenameToReadFrom.str();
		char *fileName = (char*) tmp.c_str();
		inputfiles[i] = fopen(fileName, "r");

		std::stringstream filenameToReadFromSize;
		filenameToReadFromSize << final_file_prefix << world.rank() << "_" << i << "_size.dat";
		tmp = filenameToReadFromSize.str();
		fileName = (char*) tmp.c_str();
		FILE* tmpFile = fopen(fileName, "r");
		fscanf(tmpFile, "%d\n", &nnzInFiles[i]);
		total_nnz += nnzInFiles[i];
		fclose(tmpFile);
	}
	std::vector<L> receive_COO_row_id(total_nnz, 0);
	std::vector<L> receive_COO_col_id(total_nnz, 0);
	std::vector<D> receive_COO_vals(total_nnz, 0);
	total_nnz = 0;

	for (int file = 0; file < world.size(); file++) {
		for (int i = 0; i < nnzInFiles[file]; i++) {
			fscanf(inputfiles[file], "%d,%d,%f", &receive_COO_row_id[total_nnz], &receive_COO_col_id[total_nnz],
					&receive_COO_vals[total_nnz]);
			total_nnz++;
		}
	}
	for (int i = 0; i < world.size(); i++) {
		fclose(inputfiles[i]);
	}

	if (world.rank() == 0) {
		//load sizes
		std::stringstream filenameToReadFromSize;
		filenameToReadFromSize << final_file_prefix << "desc.dat";
		string tmp = filenameToReadFromSize.str();
		char * fileName = (char*) tmp.c_str();
		FILE* tmpFile = fopen(fileName, "r");
		fscanf(tmpFile, "m:%d\nn:%d\n", &some_columns.m, &some_columns.total_n);
		fclose(tmpFile);

		std::stringstream filenameToReadFromSizeB;
		filenameToReadFromSizeB << final_file_prefix << "b.dat";
		tmp = filenameToReadFromSizeB.str();
		fileName = (char*) tmp.c_str();
		tmpFile = fopen(fileName, "r");
		some_columns.b.resize(some_columns.m);
		for (int i = 0; i < some_columns.m; i++) {
			fscanf(tmpFile, "%f", &some_columns.b[i]);
		}
		fclose(tmpFile);
	}
	broadcast(world, some_columns.b, 0);
	broadcast(world, some_columns.m, 0);
	broadcast(world, some_columns.total_n, 0);


	//TODO CHANGE FROM COO TO CSC
	getCSC_from_COO_plus_find_size(receive_COO_vals, receive_COO_row_id, receive_COO_col_id,
				some_columns.A_csc_values, some_columns.A_csc_row_idx, some_columns.A_csc_col_ptr, some_columns.m,
				some_columns.n, dataDistributor, world.rank());


}

template<typename D, typename L>
void create_distributed_partitions(mpi::communicator &world, data_distributor<L, D> &dataDistributor,
		ProblemData<L, D> &some_rows, ProblemData<L, D> &some_columns, DistributedSettings &settings,
		distributed_statistics &stat, string final_file_prefix) {
	// Each node has its data initially in some_rows, and tries to move them to some_columns
	all_reduce(world, (long) some_rows.n, stat.instance_columns, mpi::maximum<long>());
	// compute row shifts
	std::vector<L> ms(world.size(), 0);
	obtain_row_shift_data(ms, world, some_rows, dataDistributor);
	dataDistributor.columns_parts.resize(stat.instance_columns, 0);
	L total_lenght = dataDistributor.columns_parts.size();
	L total_lenght_per_partition = total_lenght / world.size() + 1;
	for (int i = 0; i < total_lenght; i++) {
		dataDistributor.columns_parts[i] = i / total_lenght_per_partition;
	}
	long instance_nnz_local = some_rows.A_csr_values.size();
	all_reduce(world, instance_nnz_local, stat.instance_nnz, std::plus<long>());
	compute_distribution_partitions(world, dataDistributor, some_rows, settings, stat, ms);
	// ============= PART 2 ================
	FILE* outputfiles[world.size()];
	for (int i = 0; i < world.size(); i++) {
		std::stringstream filenameToWrite;
		filenameToWrite << final_file_prefix << i << "_" << world.rank() << ".dat";
		string tmp = filenameToWrite.str();
		char *fileName = (char*) tmp.c_str();
		outputfiles[i] = fopen(fileName, "w+");
	}
	std::vector<int> rows_of_file(world.size(), 0);
	for (L row = 0; row < some_rows.m; row++) {
		for (L tmp = some_rows.A_csr_row_ptr[row]; tmp < some_rows.A_csr_row_ptr[row + 1]; tmp++) {
			L col = some_rows.A_csr_col_idx[tmp];
			L part = dataDistributor.columns_parts[col];
			fprintf(outputfiles[part], "%d,%d,%1.16f\n", row + ms[world.rank()], col, some_rows.A_csr_values[tmp]);
			rows_of_file[part]++;
		}
	}

	for (int i = 0; i < world.size(); i++) {
		std::stringstream filenameToWrite;
		filenameToWrite << final_file_prefix << i << "_" << world.rank() << "_size.dat";
		string tmp = filenameToWrite.str();
		char *fileName = (char*) tmp.c_str();
		FILE* tmpFile = fopen(fileName, "w+");
		fprintf(tmpFile, "%d\n", rows_of_file[i]);
		fclose(tmpFile);

		fclose(outputfiles[i]);
	}










	all_reduce(world, (long) some_rows.m, stat.instance_rows, std::plus<long>());
	some_columns.m = stat.instance_rows;
	some_rows.n = stat.instance_columns;
	// Obtain "b" vector
	std::vector < std::vector<D> > all_partial_b;
	gather(world, some_rows.b, all_partial_b, 0);
	some_columns.b.resize(some_columns.m);
	if (world.rank() == 0) {
		std::stringstream filenameToWrite;
		filenameToWrite << final_file_prefix << "b" << ".dat";
		string tmp = filenameToWrite.str();
		char *fileName = (char*) tmp.c_str();
		FILE * bfile = fopen(fileName, "w+");

		L pos = 0;
		for (L partId = 0; partId < all_partial_b.size(); partId++) {
			for (L j = 0; j < all_partial_b[partId].size(); j++) {
				fprintf(bfile, "%1.16f\n", all_partial_b[partId][j]);
			}
		}
		fclose(bfile);
		std::stringstream filenameToWrite2;
		filenameToWrite2 << final_file_prefix << "desc" << ".dat";
		tmp = filenameToWrite2.str();
		fileName = (char*) tmp.c_str();
		FILE * descriptionFile = fopen(fileName, "w+");
		fprintf(descriptionFile, "m:%d\nn:%d\n", some_columns.m, stat.instance_columns);
		fclose(descriptionFile);
	}
	//	broadcast(world, some_columns.b, 0);
	//	some_columns.x.resize(some_columns.n, 0);
	// compute the imbalance
	//	long max_columns = 0;
	//	long min_columns = 0;
	//	all_reduce(world, (long) some_columns.n, max_columns, mpi::maximum<long>());
	//	all_reduce(world, (long) some_columns.n, min_columns, mpi::minimum<long>());
	//	stat.imbalance_columns = 1.0 * min_columns / max_columns;
	//	long max_nnz = 0;
	//	long min_nnz = 0;
	//	all_reduce(world, (long) some_columns.A_csc_values.size(), max_nnz, mpi::maximum<long>());
	//	all_reduce(world, (long) some_columns.A_csc_values.size(), min_nnz, mpi::minimum<long>());
	//	stat.imbalance_nnz = 1.0 * min_columns / (1.0 * max_columns);

}

template<typename D, typename L>
void distribute_data_from_nontrivial_sources(mpi::communicator &world,
		data_distributor<L, D> &dataDistributor, ProblemData<L, D> &some_rows, ProblemData<L, D> &some_columns,
		DistributedSettings &settings, distributed_statistics &stat) {
	// Each node has its data initially in some_rows, and tries to move them to some_columns
	all_reduce(world, (long) some_rows.n, stat.instance_columns, mpi::maximum<long>());
	// compute row shifts
	std::vector<L> ms(world.size(), 0);
	obtain_row_shift_data(ms, world, some_rows, dataDistributor);
	dataDistributor.columns_parts.resize(stat.instance_columns, 0);
	L total_lenght = dataDistributor.columns_parts.size();
	L total_lenght_per_partition = total_lenght / world.size() + 1;
	for (int i = 0; i < total_lenght; i++) {
		dataDistributor.columns_parts[i] = i / total_lenght_per_partition;
	}
	long instance_nnz_local = some_rows.A_csr_values.size();
	all_reduce(world, instance_nnz_local, stat.instance_nnz, std::plus<long>());
	// compute distributions into parts
	compute_distribution_partitions(world, dataDistributor, some_rows, settings, stat, ms);
	std::vector<L> receive_COO_row_id;
	std::vector<L> receive_COO_col_id;
	std::vector<D> receive_COO_vals;
	// exchange data between PC and fill my data in COO format
	exchange_data_and_fill_my_part_data_in_COO(receive_COO_row_id, receive_COO_col_id, receive_COO_vals, ms,
			world, some_rows, dataDistributor);
	// conver my data from COO into CSC
	getCSC_from_COO_plus_find_size(receive_COO_vals, receive_COO_row_id, receive_COO_col_id,
			some_columns.A_csc_values, some_columns.A_csc_row_idx, some_columns.A_csc_col_ptr, some_columns.m,
			some_columns.n, dataDistributor, world.rank());
	// obtain the true "m"
	all_reduce(world, (long) some_rows.m, stat.instance_rows, std::plus<long>());
	some_columns.m = stat.instance_rows;
	some_rows.n = stat.instance_columns;
	// Obtain "b" vector
	std::vector < std::vector<D> > all_partial_b;
	gather(world, some_rows.b, all_partial_b, 0);
	some_columns.b.resize(some_columns.m);
	if (world.rank() == 0) {
		L pos = 0;
		for (L partId = 0; partId < all_partial_b.size(); partId++) {
			for (L j = 0; j < all_partial_b[partId].size(); j++) {
				some_columns.b[pos] = all_partial_b[partId][j];
				pos++;
			}
		}
	}
	broadcast(world, some_columns.b, 0);
	some_columns.x.resize(some_columns.n, 0);
	// compute the imbalance
	long max_columns = 0;
	long min_columns = 0;
	all_reduce(world, (long) some_columns.n, max_columns, mpi::maximum<long>());
	all_reduce(world, (long) some_columns.n, min_columns, mpi::minimum<long>());
	stat.imbalance_columns = 1.0 * min_columns / max_columns;
	long max_nnz = 0;
	long min_nnz = 0;
	all_reduce(world, (long) some_columns.A_csc_values.size(), max_nnz, mpi::maximum<long>());
	all_reduce(world, (long) some_columns.A_csc_values.size(), min_nnz, mpi::minimum<long>());
	stat.imbalance_nnz = 1.0 * min_columns / (1.0 * max_columns);
}

#endif
