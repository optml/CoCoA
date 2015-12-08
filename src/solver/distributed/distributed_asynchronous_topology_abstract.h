/*
 * distributed_asynchronous_topology_abstract.h
 *
 *  Created on: 18 May 2012
 *      Author: jmarecek
 */

#ifndef DISTRIBUTED_ASYNCHRONOUS_TOPOLOGY_ABSTRACT_H_
#define DISTRIBUTED_ASYNCHRONOUS_TOPOLOGY_ABSTRACT_H_

// "Traits" defining what loss functions to use
struct cluster_topology_indexing {
};

template<typename L, typename Traits>
class Topology {
public:

	static inline int next_rung_index(L world_size, L world_rank, L width, L index) {
		return 0;
	}

	static inline int previous_rung_index(L world_size, L world_rank, L width, L index) {
		return 0;
	}

	static inline int count_rungs(L world_size, L width) {
		return 0;
	}

	static inline int previous_rung(L world_size, L world_rank, L width) {
		return 0;
	}

	static inline int this_rung(L world_size, L world_rank, L width) {
		return 0;
	}

	static inline int this_rung_index(L world_size, L world_rank, L width, L index) {
		return 0;
	}
};


#endif /* DISTRIBUTED_ASYNCHRONOUS_TOPOLOGY_H_ */
