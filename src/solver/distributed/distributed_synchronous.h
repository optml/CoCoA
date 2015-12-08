#ifndef ACDC_DISTRIBUTED_SYNCHRONOUS
#define ACDC_DISTRIBUTED_SYNCHRONOUS

#include <set>
#include <vector>
#include <numeric>

#include <algorithm>
#include <iterator>

#include <omp.h>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include "distributed_essentials.h"

#include "distributed_include.h"

//#include "solver/l2l1/parallel/random_solver_nonoverlapping.h"

#include "distributed_structures.h"

using namespace std;
namespace mpi = boost::mpi;

template<typename D, typename L, typename I>
void shift_residuals_point_to_point(mpi::environment &env,
		mpi::communicator &world, data_distributor<L, D> &dataDistributor,
		std::vector<D> &residuals, std::vector<D> &residual_updates, I round) {
	int r = 0;
	std::vector<mpi::request> reqs(2 * dataDistributor.talking_members.size());

	int tag = 0;

	// at first do the coordinate for reduce:
	if (dataDistributor.coordinates_of_reduce.size() > 0) {
		for (int i = 0; i < dataDistributor.coordinates_of_reduce.size(); i++) {
			dataDistributor.residual_update_buffer_reduce_sent[i] =
					residual_updates[dataDistributor.coordinates_of_reduce[i]];
		}
		all_reduce(world, dataDistributor.residual_update_buffer_reduce_sent,
				dataDistributor.residual_update_buffer_reduce_receive,
				vectorplus<D>());
		// do the update
		for (int i = 0; i < dataDistributor.coordinates_of_reduce.size(); i++) {
			residuals[dataDistributor.coordinates_of_reduce[i]] +=
					dataDistributor.residual_update_buffer_reduce_receive[i];
		}
	}
	// do other comunication
	for (typename std::set<L>::iterator talk_to =
			dataDistributor.talking_members.begin();
			talk_to != dataDistributor.talking_members.end(); talk_to++) {
		// create data for "talk_to"
		L j = 0;
		for (typename std::list<L>::iterator row_coordinate =
				dataDistributor.coordinates_of_updates[*talk_to].begin();
				row_coordinate
						!= dataDistributor.coordinates_of_updates[*talk_to].end();
				row_coordinate++) {
			dataDistributor.residual_update_buffer_sent[*talk_to][j] =
					residual_updates[*row_coordinate];
			j++;
		}
		int talk_to_int = *talk_to;
		reqs[r++] = visend(world, talk_to_int, tag,
				dataDistributor.residual_update_buffer_sent[*talk_to]);
		reqs[r++] = virecv(world, talk_to_int, tag,
				dataDistributor.residual_update_buffer_receive[*talk_to]);
	}

	L j = 0;
	for (typename std::list<L>::iterator row_coordinate =
			dataDistributor.coordinates_of_updates[world.rank()].begin();
			row_coordinate
					!= dataDistributor.coordinates_of_updates[world.rank()].end();
			row_coordinate++) {
		residuals[*row_coordinate] += residual_updates[*row_coordinate];
	}

	mpi::wait_all(&reqs[0], &reqs[r]);
	// Update my residuals
	for (typename std::set<L>::iterator talk_to =
			dataDistributor.talking_members.begin();
			talk_to != dataDistributor.talking_members.end(); talk_to++) {
		// create data for "talk_to"
		L j = 0;
		for (typename std::list<L>::iterator row_coordinate =
				dataDistributor.coordinates_of_updates[*talk_to].begin();
				row_coordinate
						!= dataDistributor.coordinates_of_updates[*talk_to].end();
				row_coordinate++) {
			residuals[*row_coordinate] +=
					dataDistributor.residual_update_buffer_receive[*talk_to][j];
			j++;
		}
	}

	//
	//
	//	typename std::vector<D> empty_v(residuals.size(), 0);
	//	typename std::vector<std::vector<D> > received(talk_to.size(), empty_v);
	//
	//	int incoming = 0;
	//	typename std::list<L>::iterator it;
	//	for (it = talk_to.begin(); it != talk_to.end(); it++) {
	//		int other = *it;
	//		// std::cout << world.rank() << " talks to " << other << std::endl;
	//
	//	}
	//
	//	// broadcasting
	//	for (it = broadcasting.begin(); it != broadcasting.end(); it++) {
	//		if (*it == world.rank()) {
	//			// std::cout << "BROADCASTING FROM " << world.rank() << std::endl;
	//			broadcast(world, residual_updates, world.rank());
	//		} else {
	//			// std::cout << "RECEIVING BROADCAST FROM " << *it << std::endl;
	//			std::vector<D> broadcasted(residuals.size(), 0);
	//			broadcast(world, broadcasted, *it);
	//			for (int i = 0; i < residuals.size(); i++) {
	//				residuals[i] += broadcasted[i];
	//			}
	//		}
	//	}
	//
	//	// Instead of waiting at the barrier, we can update the residuals with our updates
	//	int i;
	//	for (i = 0; i < residuals.size(); i++) {
	//		residuals[i] += residual_updates[i];
	//	}
	//
	//	if (r > 0)
	//
	//
	//	// Update with whatever was sent point-to-point
	//	int update;
	//	for (update = 0; update < talk_to.size(); update++)
	//		for (i = 0; i < residuals.size(); i++) {
	//			residuals[i] += received[update][i];
	//		}

}

template<typename D, typename L>
void shift_residuals_point_to_point(mpi::environment &env,
		mpi::communicator &world, std::list<L> & talk_to,
		std::list<L> & broadcasting, std::vector<D> &residuals,
		std::vector<D> &residual_updates, L round) {

	int tag = round % env.max_tag();

	int r = 0;
	mpi::request reqs[2 * talk_to.size()];

	typename std::vector<D> empty_v(residuals.size(), 0);
	typename std::vector<std::vector<D> > received(talk_to.size(), empty_v);

	int incoming = 0;
	typename std::list<L>::iterator it;
	for (it = talk_to.begin(); it != talk_to.end(); it++) {
		int other = *it;
		// std::cout << world.rank() << " talks to " << other << std::endl;
		reqs[r++] = visend(world, other, tag, residual_updates);
		reqs[r++] = virecv(world, other, tag, received[incoming++]);
	}

	// broadcasting
	for (it = broadcasting.begin(); it != broadcasting.end(); it++) {
		if (*it == world.rank()) {
			// std::cout << "BROADCASTING FROM " << world.rank() << std::endl;
			boost::mpi::broadcast(world, residual_updates, world.rank());
		} else {
			// std::cout << "RECEIVING BROADCAST FROM " << *it << std::endl;
			std::vector<D> broadcasted(residuals.size(), 0);
			boost::mpi::broadcast(world, broadcasted, *it);
			for (int i = 0; i < residuals.size(); i++) {
				residuals[i] += broadcasted[i];
			}
		}
	}

	// Instead of waiting at the barrier, we can update the residuals with our updates
	int i;
	for (i = 0; i < residuals.size(); i++) {
		residuals[i] += residual_updates[i];
	}

	if (r > 0)
		mpi::wait_all(reqs, reqs + r);

	// Update with whatever was sent point-to-point
	int update;
	for (update = 0; update < talk_to.size(); update++)
		for (i = 0; i < residuals.size(); i++) {
			residuals[i] += received[update][i];
		}

}

template<typename D, typename L>
void shift_residuals_sparse(mpi::environment &env, mpi::communicator &world,
		data_distributor<L, D> &dataDistributor, std::vector<D> &residuals,
		std::vector<D> &residual_updates, L round) {
	/*
	 int tag = round % env.max_tag();

	 int r = 0;
	 mpi::request reqs[2 * talk_to.size()];

	 typename std::list<L>::iterator with_whom;
	 typename std::list<std::list<L> >::iterator about_what;

	 typename std::vector<std::list<D> > received;
	 typename std::vector<std::list<D> > sending;

	 // we are looping over two lists
	 with_whom = talk_to.begin();
	 about_what = what_to_exchange.begin();
	 for (; with_whom != talk_to.end();) {
	 // initialise the structures for sending data
	 typename std::list<D> to_send;
	 typename std::list<L>::iterator coef;
	 for (coef = about_what->begin(); coef != about_what->end(); coef++) {
	 to_send.push_back(residual_updates[*coef]);
	 }
	 sending.push_back(to_send);
	 // initialise the structures for receiving data
	 std::list<D> dummy(about_what->size(), 0);
	 received.push_back(dummy);
	 with_whom++;
	 about_what++;
	 }

	 int i = 0;
	 with_whom = talk_to.begin();
	 for (; with_whom != talk_to.end();) {
	 // std::cout << "p2p: " << world.rank() << " " << *with_whom << " " << sending[i].size() << " " << received[i].size() << std::endl;
	 // NOTE: These are not vectors of doubles, hence best not replaced with visend
	 reqs[r++] = world.isend(*with_whom, tag, sending[i]);
	 reqs[r++] = world.irecv(*with_whom, tag, received[i]);
	 i += 1;
	 with_whom++;
	 }

	 // broadcasting
	 with_whom = broadcasting.begin();
	 about_what = what_to_broadcast.begin();
	 for (; with_whom != broadcasting.end();) {
	 typename std::list<D> to_broadcast;
	 typename std::list<L>::iterator coef;
	 if (*with_whom == world.rank()) {
	 for (coef = about_what->begin(); coef != about_what->end(); coef++) {
	 to_broadcast.push_back(residual_updates[*coef]);
	 }
	 boost::mpi::broadcast(world, to_broadcast, world.rank());
	 } else {
	 std::list<D> broadcasted(about_what->size(), 0);
	 boost::mpi::broadcast(world, broadcasted, *with_whom);
	 typename std::list<D>::iterator in_broadcasted = broadcasted.begin();
	 for (coef = about_what->begin(); coef != about_what->end(); coef++) {
	 residuals[*coef] += *in_broadcasted;
	 in_broadcasted++;
	 }
	 }
	 with_whom++;
	 about_what++;
	 }

	 // Instead of waiting at the barrier, we can update the residuals with our updates
	 for (int i = 0; i < residuals.size(); i++) {
	 residuals[i] += residual_updates[i];
	 }

	 if (r > 0)
	 mpi::wait_all(reqs, reqs + r);

	 // what we are looping over:
	 with_whom = talk_to.begin();
	 about_what = what_to_exchange.begin();
	 for (i = 0; i < received.size(); i++) {
	 // DEBUG: std::cout << world.rank() << " talks to " << other << std::endl;
	 // Now we have to apply the updates by recalling what does received[i] mean
	 typename std::list<L>::iterator coef = about_what->begin();
	 typename std::list<D>::iterator in_received = received[i].begin();
	 for (; coef != about_what->end();) {
	 residuals[*coef] += *in_received;
	 coef++;
	 in_received++;
	 }
	 }
	 */
}

template<typename D, typename L, typename I>
void shift_residuals_supersparse(mpi::environment &env,
		mpi::communicator &world, data_distributor<L, D> &dataDistributor,
		std::vector<D> &residuals, std::vector<D> &residual_updates,
		DistributedSettings &settings, I round) {
	/*
	 int tag = 2 * round % env.max_tag();

	 int r = 0;
	 mpi::request reqs[4 * talk_to.size()];

	 typename std::list<L>::iterator with_whom;
	 typename std::list<std::list<L> >::iterator about_what;

	 // initialise the structures for receiving data
	 std::vector < std::list<D> > received(talk_to.size());
	 std::vector < std::list<L> > received_manifests(talk_to.size());
	 std::vector < std::list<D> > sending;
	 std::vector < std::list<L> > sending_manifests;

	 // we are looping over two lists:
	 with_whom = talk_to.begin();
	 about_what = what_to_exchange.begin();
	 for (; with_whom != talk_to.end();) {
	 // figure out what to send
	 std::list<D> to_send;
	 std::list<L> to_send_manifest;
	 typename std::list<L>::iterator coef;
	 for (coef = about_what->begin(); coef != about_what->end(); coef++) {
	 if (residual_updates[*coef] > settings.supersparse_epsilon) {
	 to_send.push_back(residual_updates[*coef]);
	 to_send_manifest.push_back(*coef);
	 }
	 }
	 sending.push_back(to_send);
	 sending_manifests.push_back(to_send_manifest);
	 with_whom++;
	 about_what++;
	 }

	 int i = 0;
	 with_whom = talk_to.begin();
	 for (; with_whom != talk_to.end();) {
	 // NOTE: This is not vectors of doubles, hence best not overwritten with visend
	 reqs[r++] = world.isend(*with_whom, tag, sending[i]);
	 reqs[r++] = world.isend(*with_whom, tag + 1, sending_manifests[i]);
	 reqs[r++] = world.irecv(*with_whom, tag, received[i]);
	 reqs[r++] = world.irecv(*with_whom, tag + 1, received_manifests[i]);
	 with_whom++;
	 i += 1;
	 }

	 // broadcasting
	 with_whom = broadcasting.begin();
	 about_what = what_to_broadcast.begin();
	 for (; with_whom != broadcasting.end();) {
	 // NOTE: This is not correct English, but broadcast is in use by boost::mpi
	 typename std::list<D> broadcasted;
	 typename std::list<L> broadcast_manifest;
	 typename std::list<L>::iterator coef;
	 if (*with_whom == world.rank()) {
	 for (coef = about_what->begin(); coef != about_what->end(); coef++) {
	 if (residual_updates[*coef] > settings.supersparse_epsilon) {
	 broadcasted.push_back(residual_updates[*coef]);
	 broadcast_manifest.push_back(*coef);
	 }
	 }
	 boost::mpi::broadcast(world, broadcast_manifest, world.rank());
	 boost::mpi::broadcast(world, broadcasted, world.rank());
	 } else {
	 boost::mpi::broadcast(world, broadcast_manifest, *with_whom);
	 boost::mpi::broadcast(world, broadcasted, *with_whom);
	 typename std::list<D>::iterator in_broadcast = broadcasted.begin();
	 typename std::list<L>::iterator in_manifest = broadcast_manifest.begin();
	 for (; in_broadcast != broadcasted.end();) {
	 residuals[*in_manifest] += *in_broadcast;
	 in_broadcast++;
	 in_manifest++;
	 }
	 }
	 with_whom++;
	 about_what++;
	 }

	 // Instead of waiting at the barrier, we can update the residuals with our updates
	 for (i = 0; i < residuals.size(); i++) {
	 residuals[i] += residual_updates[i];
	 }

	 boost::mpi::wait_all(reqs, reqs + r);

	 // what we are looping over:
	 with_whom = talk_to.begin();
	 about_what = what_to_exchange.begin();

	 for (i = 0; i < received.size(); i++) {
	 // Now we have to apply the updates by recalling what does received[i] mean
	 typename std::list<D>::iterator in_received = received[i].begin();
	 typename std::list<L>::iterator in_manifest = received_manifests[i].begin();
	 for (; in_received != received[i].end();) {
	 residuals[*in_manifest] += *in_received;
	 in_received++;
	 in_manifest++;
	 }
	 }
	 */
}

template<typename D, typename L>
void gather_residuals(mpi::communicator &world, std::vector<D> &residuals,
		std::vector<D> &residual_updates, ProblemData<L, D> &part) {

	D sum_of_residuals = 0;
	if (world.rank() == 0) {
		std::vector<D> empty_v(part.m, 0);
		std::vector < std::vector<D> > all_parts(world.size(), empty_v);
		boost::mpi::gather(world, residual_updates, all_parts, 0);
		for (L i = 0; i < part.m; i++) {
			/* TODO: CHECK WHY THIS FAILS:
			 printf("%d (%d), %d (%d) \n",
			 all_parts[i].size(), residual_updates.size(),
			 std::accumulate(all_parts[i].begin(), all_parts[i].end(), 0),
			 std::accumulate(residual_updates.begin(), residual_updates.end(), 0)
			 );
			 */
		}
		for (L i = 0; i < part.m; i++) {
			for (L j = 1; j < world.size(); j++) {
				all_parts[0][i] += all_parts[j][i];
			}
			residuals[i] += all_parts[0][i];
			sum_of_residuals += residuals[i] * residuals[i];
		}
		boost::mpi::broadcast(world, all_parts[0], 0);
	} else {
		boost::mpi::gather(world, residual_updates, 0);
		boost::mpi::broadcast(world, residual_updates, 0);
		for (L i = 0; i < part.m; i++) {
			residuals[i] += residual_updates[i];
			sum_of_residuals += residuals[i] * residuals[i];
		}
	}
}

template<typename D>
inline void reduce_residuals(mpi::communicator &world,
		std::vector<D> &residuals, std::vector<D> &residual_updates,
		std::vector<D> &buffer) {
	vall_reduce(world, residual_updates, buffer);

	cblas_sum_of_vectors(residuals, buffer);

}

#endif // ACDC_DISTRIBUTED_SYNCHRONOUS
