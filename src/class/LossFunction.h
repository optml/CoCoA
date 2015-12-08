/*
 * LossFunction.h
 *
 *  Created on: Dec 1, 2014
 *      Author: taki
 */

#ifndef LOSSFUNCTION_H_
#define LOSSFUNCTION_H_

template<typename L, typename D>
class LossFunction {
public:
	LossFunction(){

	}
	virtual ~LossFunction() {}

	virtual void init(ProblemData<L, D> & instance){
	}

	virtual void computeObjectiveValue(ProblemData<L, D> & instance,
			mpi::communicator & world, std::vector<D> & w, double &finalDualError,
			double &finalPrimalError){

	}

	virtual void subproblem_solver_SDCA(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
			std::vector<D> &w, std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
			mpi::communicator &world, D gamma, Context &ctx, std::ofstream &logFile) {

	}

	virtual void accelerated_SDCA_oneIteration(ProblemData<L, D> &instance,
			std::vector<D> &deltaAlpha, std::vector<D> &w, std::vector<D> &deltaW,
			std::vector<D> &zk, std::vector<D> &uk,
			std::vector<D> &yk, std::vector<D> &deltayk, std::vector<D> &Ayk, std::vector<D> &deltaAyk,
			D &theta, DistributedSettings & distributedSettings){

	}

    virtual void subproblem_solver_SDCA_without_duality(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
      std::vector<D> &w, std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
      mpi::communicator &world, D gamma, Context &ctx, std::ofstream &logFile) {

     }

	virtual void subproblem_solver_accelerated_SDCA(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
			std::vector<D> &w, std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
			mpi::communicator &world, D gamma, Context &ctx, std::ofstream &logFile) {

	}

	virtual void subproblem_solver_steepestdescent(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
			std::vector<D> &w, std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
			mpi::communicator &world, D gamma, Context &ctx, std::ofstream &logFile) {

	}

	virtual void subproblem_solver_CG(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
			std::vector<D> &w, std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
			mpi::communicator &world, D gamma, Context &ctx, std::ofstream &logFile) {

	}

	virtual void subproblem_solver_LBFGS(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
			std::vector<D> &w, std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
			mpi::communicator &world, D gamma, Context &ctx, std::ofstream &logFile) {

	}

	virtual void subproblem_solver_BB(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
			std::vector<D> &w, std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
			mpi::communicator &world, D gamma, Context &ctx, std::ofstream &logFile) {

	}

	virtual void subproblem_solver_FISTA(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
			std::vector<D> &w, std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
			mpi::communicator &world, D gamma, Context &ctx, std::ofstream &logFile) {

	}

	virtual void LBFGS_update(ProblemData<L, D> &instance, std::vector<D> &search_direction, std::vector<D> &old_grad,
			std::vector<D> &sk, std::vector<D> &rk, std::vector<D> &gradient, std::vector<D> &oneoversy,
			L iter_counter, int limit_BFGS, int flag_BFGS) {

	}

	virtual void backtrack_linesearch(ProblemData<L, D> &instance,
			std::vector<D> &deltaAlpha, std::vector<D> &search_direction, std::vector<D> &w, D dualobj,
			D &rho, D &c1ls, D &a, DistributedSettings & distributedSettings){

	}

	virtual void compute_subproproblem_obj(ProblemData<L, D> &instance,
			std::vector<D> &deltaAlpha, std::vector<D> &search_direction, std::vector<D> &w, D dualobj,
			D &rho, D &c1ls, D &a, DistributedSettings & distributedSettings){

	}
	virtual void compute_subproproblem_gradient(ProblemData<L, D> &instance,
			std::vector<D> &gradient, std::vector<D> &deltaAlpha, std::vector<D> &w){

	}

};

#endif /* LOSSFUNCTION_H_ */
