#include <agency/agency.hpp>
#include <numeric>
#include <iostream>


thread_local int g_thread_index = -1;


// XXX I changed this to sum integers to make it easier to validate its result,
//     but this approach is perfectly applicable to floating point as well
class AgencyReduceSum
{
public:

	//
	// Constructor takes default value (default ctor is disabled).
	//
	AgencyReduceSum(double init_val, int num_threads = 1)
          : m_num_threads(num_threads)
  	{
		m_is_copy = false;

		m_init_val = init_val;
		m_reduced_val = 0.0;

		m_partial_sum = new double[m_num_threads];


		for(int i = 0; i < m_num_threads; ++i){
		  m_partial_sum[i] = init_val;
		}

	}

	//
	// Copy ctor.
	//
	AgencyReduceSum(const AgencyReduceSum& other)
	{
		*this = other;
		m_is_copy = true;

	}

	//
	//
	~AgencyReduceSum()
	{
		if (!m_is_copy) {
			delete m_partial_sum;
		}

	}

	//
	// Operator that returns reduced sum value.
	//
	operator double()
	{
		double tmp_reduced_val = 0.0;
		for (size_t i = 0; i < m_num_threads; ++i) {
			tmp_reduced_val += m_partial_sum[i];
		}
		m_reduced_val = m_init_val + tmp_reduced_val;

		return m_reduced_val;
	}

	//
	// Method that returns sum value.
	//
	int get() { return operator double(); }

	//
	// += operator that adds value to sum for current thread.
	//
	AgencyReduceSum operator+=(double val) const
	{

		//std::cout << tid << std::endl;

		m_partial_sum[g_thread_index] += val;
		return *this;
	}

	private:
		//
		// Default ctor is declared private and not implemented.
		//
		AgencyReduceSum();

		bool m_is_copy;

		double m_init_val;
		double m_reduced_val;

		double* m_partial_sum;
        int m_num_threads;
};


