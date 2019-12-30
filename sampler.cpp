#include <iostream>
#include <algorithm>
#include <queue>
#include <utility>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <chrono>
#include <random>
#include <cmath>
//#define _DEBUG


class RandomEngine {
public:
  /*! \brief Constructor with default seed */
  RandomEngine() {
    rng_.seed(42);
  }

  /*!
  * \brief Generate a uniform random integer in [lower, upper)
  */
  template<typename T>
  T RandInt(T lower, T upper) {
    std::uniform_int_distribution<T> dist(lower, upper - 1);
    return dist(rng_);
  }

  /*!
  * \brief Generate a uniform random float in [lower, upper)
  */
  template<typename T>
  T Uniform(T lower, T upper) {
    std::uniform_real_distribution<T> dist(lower, upper);
    return dist(rng_);
  }
private:
  std::mt19937 rng_;
};

template <
  typename Idx,
  typename DType,
  bool replace = true>
class AliasSampler {
private:
  RandomEngine *re;
  Idx N;
  DType accum, taken;        // accumulated likelihood
  std::vector<Idx> K;             // alias table
  std::vector<DType> U;           // probability table
  std::vector<DType> _prob;       // category distribution
  std::vector<bool> used;         // indicate whether the given category has been sampled, activated when replace=false;
  std::vector<Idx> id_mapping;

  inline Idx map(Idx x) const{
    if (replace)
      return x;
    else
      return id_mapping[x];
  }

  void rebuild(const std::vector<DType>& prob) {
    N = 0;
    accum = 0.;
    taken = 0.;
    if (!replace)
      id_mapping.clear();
    for (Idx i = 0; i < prob.size(); ++i)
      if (!used[i]) {
        N++;
        accum += prob[i];
        if (!replace)
          id_mapping.push_back(i);
      }
//        if (N == 0) LOG(FATAL) << "Cannot take more sample than population when 'replace=false'";
    if (N == 0) assert(false);
    K.resize(N);
    U.resize(N);
    DType avg = accum / static_cast<DType>(N);
    std::fill(U.begin(), U.end(), avg); // initialize U
    std::queue<std::pair<Idx, DType> > under, over;
    for (Idx i = 0; i < N; ++i) {
      DType p = prob[map(i)];
      if (p > avg)
        over.push(std::make_pair(i, p));
      else
        under.push(std::make_pair(i, p));
      K[i] = i;                       // initialize K
    }
    while (!under.empty() && !over.empty()) {
      auto u_pair = under.front(), o_pair = over.front();
      Idx i_u = u_pair.first, i_o = o_pair.first;
      DType p_u = u_pair.second, p_o = o_pair.second;
      K[i_u] = i_o;
      U[i_u] = p_u;
      if (p_o + p_u > 2 * avg)
        over.push(std::make_pair(i_o, p_o + p_u - avg));
      else if (p_o + p_u < 2 * avg)
        under.push(std::make_pair(i_o, p_o + p_u - avg));
      under.pop();
      over.pop();
    }
  }

public:
  void reinit_state(const std::vector<DType>& prob) {
    used.resize(prob.size());
    if (!replace)
      _prob = prob;
    std::fill(used.begin(), used.end(), false);
    rebuild(prob);
  }

  AliasSampler(RandomEngine* re, const std::vector<DType>& prob): re(re) {
    reinit_state(prob);
  }

  ~AliasSampler() {}

  inline Idx draw() {
    DType avg = accum / N;
    if (!replace) {
      if (2 * taken >= accum)
        rebuild(_prob);
      while (true) {
        DType dice = re->Uniform<DType>(0, N); 
        Idx i = static_cast<Idx>(dice), rst;
        DType p = (dice - i) * avg, cap;
        if (p <= U[map(i)]) {
          cap = U[map(i)];
          rst = map(i);
        } else {
          cap = avg - U[map(i)];
          rst = map(K[i]);
        }
        if (!used[rst]) {
          used[rst] = true;
          taken += cap;
          return rst;
        }
      }
    }
    DType dice = re->Uniform<DType>(0, N);
    Idx i = static_cast<Idx>(dice);
    DType p = (dice - i) * avg;
    if (p <= U[map(i)]) 
        return map(i);
    else
        return map(K[i]);
  }
};

template <
  typename Idx,
  typename DType,
  bool replace = true>
class CDFSampler {
private:
  RandomEngine *re;
  Idx N;
  DType accum, taken;
  std::vector<DType> _prob;
  std::vector<DType> cdf;
  std::vector<bool> used;
  std::vector<Idx> id_mapping;

  inline Idx map(Idx x) const{
    if (replace)
      return x;
    else
      return id_mapping[x];
  }

  void rebuild(const std::vector<DType>& prob) {
    N = 0;
    accum = 0.;
    taken = 0.;
    if (!replace)
      id_mapping.clear();
    cdf.clear();
    cdf.push_back(0);
    for (Idx i = 0; i < prob.size(); ++i)
      if (!used[i]) {
        ++N;
        accum += prob[i];
        if (!replace)
          id_mapping.push_back(i);
        cdf.push_back(accum);
      }
//        if (N == 0) LOG(FATAL) << "Cannot take more sample than population when 'replace=false'";
    if (N == 0) assert(false);
  }
public:
  void reinit_state(const std::vector<DType>& prob) {
    used.resize(prob.size());
    if (!replace)
      _prob = prob;
    std::fill(used.begin(), used.end(), false);
    rebuild(prob);
  }

  CDFSampler(RandomEngine *re, const std::vector<DType>& prob): re(re) {
    reinit_state(prob);
  }

  ~CDFSampler() {}

  inline Idx draw() {
    DType eps = std::numeric_limits<DType>::min();
    if (!replace) {
      if (2 * taken >= accum)
        rebuild(_prob);
      while (true) {
        DType p = std::max(re->Uniform<DType>(0., accum), eps);
        Idx rst = map(std::lower_bound(cdf.begin(), cdf.end(), p) - cdf.begin() - 1);
        DType cap = _prob[rst];
        if (!used[rst]) {
          used[rst] = true;
          taken += cap;
          return rst;
        }
      }
    }
    DType p = std::max(re->Uniform<DType>(0., accum), eps);
    return map(std::lower_bound(cdf.begin(), cdf.end(), p) - cdf.begin() - 1);
  }
  
};

template <
  typename Idx,
  typename DType,
  bool replace = false>
class TreeSampler {
private:
  RandomEngine *re;
  std::vector<DType> weight;
  Idx N, num_leafs;
public:
  void reinit_state(const std::vector<DType>& prob) {
    std::fill(weight.begin(), weight.end(), 0);
    for (int i = 0; i < prob.size(); ++i)
      weight[num_leafs + i] = prob[i];
    for (int i = num_leafs - 1; i >= 1; --i)
      weight[i] = weight[i * 2] + weight[i * 2 + 1];
  }    

  TreeSampler(RandomEngine *re, const std::vector<DType>& prob): re(re) {
    num_leafs = 1;
    while (num_leafs < prob.size())
      num_leafs *= 2;
    N = num_leafs * 2;
    weight.resize(N);
    reinit_state(prob);
  }

  inline Idx draw() {
    Idx cur = 1;
    DType p = re->Uniform<DType>(0., weight[cur]);
    DType accum = 0.;
    while (cur < num_leafs) {
      DType pivot = accum + weight[cur * 2];
      Idx shift = static_cast<Idx>(p > pivot);
      cur = cur * 2 + shift;
      if (shift == 1)
        accum = pivot;
    }
    Idx rst = cur - num_leafs;
    if (!replace) {
      while (cur >= 1) {
        if (cur >= num_leafs)
          weight[cur] = 0.;
        else
          weight[cur] = weight[cur * 2] + weight[cur * 2 + 1];
        cur /= 2;
      }
    }
    return rst;
  }
};

const int num_categories = 10000000;
const int num_rolls = 10000000;
const bool replace = false;
std::vector<float> prob;
std::vector<int> cnt(num_categories, 0);

int main(int argc, char** argv) {
  RandomEngine re;
  using clock = std::chrono::system_clock;
  using millisec = std::chrono::duration<float, std::milli>;
  float sum_prob = 0;
  for (int i = 0; i < num_categories; ++i) {
    prob.push_back(re.Uniform<float>(0, 1));
    sum_prob += prob.back();
  }
  for (int i = 0; i < num_categories; ++i)
    prob[i] /= sum_prob;
#ifdef _DEBUG
  for (int i = 0; i < num_categories; ++i)
    std::cout << prob[i] << " ";
  std::cout << std::endl;
#endif
  auto tic = clock::now();
  AliasSampler<int, float, replace> as1(&re, prob);
  millisec dur = clock::now() - tic;
  std::cout << "Alias sampler building time: " << dur.count() << " ms" << std::endl;
  std::fill(cnt.begin(), cnt.end(), 0);
  tic = clock::now();
  for (int i = 0; i < num_rolls; ++i) {
    int dice = as1.draw();
    cnt[dice]++;
  }
  dur = clock::now() - tic;
  std::cout << "Alias sampler duration: " << dur.count() << " ms" << std::endl;
#ifdef _DEBUG
  for (int i = 0; i < num_categories; ++i)
    std::cout << cnt[i] / float(num_rolls) << " ";
  std::cout << std::endl;
#endif

  tic = clock::now();
  CDFSampler<int, float, replace> cs1(&re, prob);
  dur = clock::now() - tic;
  std::cout << "CDF sampler building time: " << dur.count() << " ms" << std::endl;
  std::fill(cnt.begin(), cnt.end(), 0);
  tic = clock::now();
  for (int i = 0; i < num_rolls; ++i) {
    int dice = cs1.draw();
    cnt[dice]++;
  }
  dur = clock::now() - tic;
  std::cout << "CDF sampler duration: " << dur.count() << " ms" << std::endl;
#ifdef _DEBUG
  for (int i = 0; i < num_categories; ++i)
    std::cout << cnt[i] / float(num_rolls) << " ";
  std::cout << std::endl;
#endif

  tic = clock::now();
  TreeSampler<int, float, replace> ts1(&re, prob);
  dur = clock::now() - tic;
  std::cout << "Tree sampler building time: " << dur.count() << " ms" << std::endl;
  std::fill(cnt.begin(), cnt.end(), 0);
  tic = clock::now();
  for (int i = 0; i < num_rolls; ++i) {
    int dice = ts1.draw();
    cnt[dice]++;
  }
  dur = clock::now() - tic;
  std::cout << "Tree sampler duration: " << dur.count() << " ms" << std::endl;
#ifdef _DEBUG
  for (int i = 0; i < num_categories; ++i)
    std::cout << cnt[i] / float(num_rolls) << " ";
  std::cout << std::endl;
#endif

  return 0;
}
