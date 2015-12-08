#ifndef ACDC_DISTRIBUTED_ESSENTIALS
#define ACDC_DISTRIBUTED_ESSENTIALS

#include <vector>
#include <algorithm>
#include <functional>
#include <iostream>

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/vector.hpp>




// #define USE_CUSTOM_VECTORS
#define USE_BOOST_BITWISE_SERIALISATION
// See below for #include <boost/serialization/vector.hpp>
// or a custom implementation for heterogenous clusters

template<typename T>
void vsend(boost::mpi::communicator &comm, int dest, int tag,
		const std::vector<T>& data) {
	comm.send(dest, tag, &data[0], data.size());
}

template<typename T>
boost::mpi::request visend(boost::mpi::communicator &comm, int dest, int tag,
		const std::vector<T>& data) {
	return comm.isend(dest, tag, &data[0], data.size());
}
template<typename T>
boost::mpi::request visend(boost::mpi::communicator &comm, int dest, int tag,
		T* data, unsigned int length) {
	return comm.isend(dest, tag, data, length);
}

template<typename T>
boost::mpi::request virecv(boost::mpi::communicator &comm, int dest, int tag,
		std::vector<T>& data) {
	return comm.irecv(dest, tag, &data[0], data.size());
}

template<typename T>
boost::mpi::request virecv(boost::mpi::communicator &comm, int dest, int tag,
		T* data, unsigned int length) {
	return comm.irecv(dest, tag, data, length);
}

template<typename T>
void vbroadcast(boost::mpi::communicator &comm, std::vector<T>& data, int tag) {
	boost::mpi::broadcast(comm, &data[0], data.size(), tag);
}

template<typename T>
void vbroadcast(boost::mpi::communicator &comm, T* data, unsigned int size,
		int tag) {
	boost::mpi::broadcast(comm, data, size, tag);
}

template<typename D> struct vectorplus: std::binary_function<std::vector<D>,
		std::vector<D>, std::vector<D> > {
	std::vector<D> operator()(const std::vector<D>& x,
			const std::vector<D>& y) const {
		std::vector<D> res(x.size(), 0);
		// NOTE: We assume the vectors are of the same length. This is just to make sure:
		const int minlen = std::min(x.size(), y.size());
		// TODO: do it in parallel
		for (int i = 0; i < minlen; i++) {
			res[i] = x[i] + y[i];
		}
		return res;
	}
};

template<typename T>
void vreduce(boost::mpi::communicator &comm, const std::vector<T>& input,
		std::vector<T>& output, int root) {
	// instead of reduce(comm, input, output, vectorplus<T>());
	boost::mpi::reduce(comm, &input[0], input.size(), &output[0],
			std::plus<T>(), root);
}

template<typename T>
void vreduce(boost::mpi::communicator &comm, const T* input, T* output,
		unsigned int size, int root) {
	// instead of reduce(comm, input, output, vectorplus<T>());
	boost::mpi::reduce(comm, input, size, output, std::plus<T>(), root);
}

template<typename T>
void vreduce_max(boost::mpi::communicator &comm, const T* input, T* output,
		unsigned int size, int root) {
	// instead of reduce(comm, input, output, vectorplus<T>());
	boost::mpi::reduce(comm, input, size, output, mpi::maximum<T>(), root);
}



template<typename T>
void vreduce_into_local(boost::mpi::communicator &comm, const unsigned int size,
		T* values, int root) {
	// instead of reduce(comm, input, output, vectorplus<T>());
	boost::mpi::reduce(comm, values, size, std::plus<T>(), root);
}

template<typename T>
void vreduce_max_into_local(boost::mpi::communicator &comm,
		const unsigned int size, T* values, int root) {
	// instead of reduce(comm, input, output, vectorplus<T>());
	boost::mpi::reduce(comm, values, size, std::plus<T>(), root);
}

template<typename T>
void vall_reduce(boost::mpi::communicator &comm, const std::vector<T>& input,
		std::vector<T>& output) {
	// instead of all_reduce(comm, input, output, vectorplus<T>());
	boost::mpi::all_reduce(comm, &input[0], input.size(), &output[0],
			std::plus<T>());
}

template<typename T>
void vall_reduce(boost::mpi::communicator &comm, T* input, T* output,
		const unsigned int size) {
	// instead of all_reduce(comm, input, output, vectorplus<T>());
	boost::mpi::all_reduce(comm, input, size, output, std::plus<T>());
}

template<typename T>
void vall_reduce_minimum(boost::mpi::communicator &comm, T* input, T* output,
		const unsigned int size) {
	// instead of all_reduce(comm, input, output, vectorplus<T>());
	boost::mpi::all_reduce(comm, input, size, output,  mpi::minimum<T>());
}
template<typename T>
void vall_reduce_maximum(boost::mpi::communicator &comm, T* input, T* output,
		const unsigned int size) {
	// instead of all_reduce(comm, input, output, vectorplus<T>());
	boost::mpi::all_reduce(comm, input, size, output,  mpi::maximum<T>());
}

namespace boost {
namespace mpi {
template<>
struct is_commutative<vectorplus<double>, std::vector<double> > : mpl::true_ {
};

template<>
struct is_commutative<vectorplus<float>, std::vector<float> > : mpl::true_ {
};

} // END boost::mpi
} // END boost

#ifdef USE_BOOST_BITWISE_SERIALISATION

namespace boost {
namespace serialization {

template<>
struct is_bitwise_serializable<std::vector<float> > : mpl::true_ {
};

template<>
struct is_bitwise_serializable<std::vector<double> > : mpl::true_ {
};

}
}

#endif // USE_BOOST_BITWISE_SERIALISATION
// we include the non-specialised templates anyway
#ifdef USE_CUSTOM_VECTORS

#define BOOST_SERIALIZATION_VECTOR_HPP

namespace boost {
	namespace serialization {

		template<class Archive, class Allocator>
		inline void save(
				Archive & ar,
				const std::vector<float, Allocator> &t,
				const unsigned int /* file_version */
		) {
			collection_size_type count (t.size());
			ar << count;
			ar.save_binary(&t[0], count*sizeof(float));
		}

		template<class Archive, class Allocator>
		inline void save(
				Archive & ar,
				const std::vector<double, Allocator> &t,
				const unsigned int /* file_version */
		) {
			collection_size_type count (t.size());
			ar << count;
			ar.save_binary(&t[0], count*sizeof(double));
		}

		template<class Archive, class Allocator>
		inline void load(
				Archive & ar,
				std::vector<float, Allocator> &t,
				const unsigned int /* file_version */
		) {
			collection_size_type count;
			ar >> count;
			if (count != t.size()) t.resize(count);
			ar.load_binary(&t[0], count*sizeof(float));
		}

		template<class Archive, class Allocator>
		inline void load(
				Archive & ar,
				std::vector<double, Allocator> &t,
				const unsigned int /* file_version */
		) {
			collection_size_type count;
			ar >> count;
			if (count != t.size()) t.resize(count);
			ar.load_binary(&t[0], count*sizeof(double));
		}

	} // END boost::serialization
} // END boost

#include <boost/serialization/collection_traits.hpp>

//BOOST_SERIALIZATION_COLLECTION_TRAITS(std::vector)

#else
#include <boost/serialization/vector.hpp>
#endif // END USE_CUSTOM_VECTORS
#endif // END ESSENTIALS
