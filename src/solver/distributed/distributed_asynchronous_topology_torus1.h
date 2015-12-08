/*
 * distributed_asynchronous_topology_torus1.h
 *
 *  Created on: 18 May 2012
 *      Author: jmarecek
 */

#ifndef DISTRIBUTED_ASYNCHRONOUS_TOPOLOGY_TORUS1_H_
#define DISTRIBUTED_ASYNCHRONOUS_TOPOLOGY_TORUS1_H_

#include "distributed_asynchronous_topology_abstract.h"

struct torus1_indexing: public cluster_topology_indexing {
};

/*************************************/
// a partial specialisation for Torus1
template<typename L>
class Topology<L, torus1_indexing> {
public:
	mpi::communicator local_rung_communicator;
//	 mpi::communicator *local_rung_communicator;
	static inline int next_rung_index(L world_size, L world_rank, L width, L index) { // indices are 0 based
		int start_this_rung = world_rank - (world_rank % width);
		// for width 1, this should be (world.rank() + 1) % (world.size());
		return (start_this_rung + width + index) % world_size;
	}

	static inline int previous_rung_index(L world_size, L world_rank, L width, L index) { // indices are 0 based
		int start_this_rung = world_rank - (world_rank % width);
		// for width = 1, this is (world.rank() - 1 + world.size()) % (world.size());
		return (start_this_rung - width + index + world_size) % world_size;
	}

	static inline int count_rungs(L world_size, L width) { // indices are 0 based
		return (world_size / width);
	}

	static inline int previous_rung(L world_size, L world_rank, L width) { // indices are 0 based
		int start_this_rung = world_rank - (world_rank % width);
		// for width = 1, this is (world.rank() - 1 + world.size()) % (world.size());
		return ((start_this_rung - width + world_size) % world_size) / width;
	}

	static inline int this_rung(L world_size, L world_rank, L width) { // indices are 0 based, strictly less than width!
		return (world_rank - (world_rank % width)) / width;
	}

	static inline int this_rung_index(L world_size, L world_rank, L width, L index) { // indices are 0 based, strictly less than width!
		int start_this_rung = world_rank - (world_rank % width);
		// if (world_rank >= start_this_rung + index) return (start_this_rung + index + 1 + world_size) % world_size;
		return (start_this_rung + index + 0 + world_size) % world_size;
	}
};

#endif /* DISTRIBUTED_ASYNCHRONOUS_TOPOLOGY_TORUS1_H_ */
