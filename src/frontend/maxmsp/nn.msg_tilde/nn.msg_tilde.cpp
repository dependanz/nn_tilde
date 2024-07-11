#include "../../../backend/backend_msg.h"
#include "../shared/circular_buffer.h"
#include "c74_min.h"
#include <chrono>
#include <semaphore>
#include <string>
#include <thread>
#include <vector>

#ifndef VERSION
#define VERSION "UNDEFINED"
#endif

using namespace c74::min;

unsigned power_ceil(unsigned x) {
  if (x <= 1)
    return 1;
  int power = 2;
  x--;
  while (x >>= 1)
    power <<= 1;
  return power;
}

class nn_msg_tilde : public object<nn_msg_tilde>, public vector_operator<> {
public:
  MIN_DESCRIPTION{"Interface for deep learning models"};
  MIN_TAGS{"audio, deep learning, ai"};
  MIN_AUTHOR{"Antoine Caillon & Axel Chemla--Romeu-Santos"};

  nn_msg_tilde(const atoms &args = {});
  ~nn_msg_tilde();

  // INLETS OUTLETS
  std::vector<std::unique_ptr<inlet<>>> m_inlets;
  std::vector<std::unique_ptr<outlet<>>> m_outlets;

  // BACKEND RELATED MEMBERS
  std::unique_ptr<Backend_MSG> m_model;
  bool m_is_backend_init = false;
  std::string m_method;
  std::vector<std::string> settable_attributes;
  bool has_settable_attribute(std::string attribute);
  c74::min::path m_path;
  int m_in_dim, m_in_ratio, m_out_dim, m_out_ratio, m_higher_ratio;

  std::vector<float> input_msg;
  std::vector<float> output_msg;

  // BUFFER RELATED MEMBERS
  int m_buffer_size;
  std::unique_ptr<circular_buffer<double, float>[]> m_in_buffer;
  std::unique_ptr<circular_buffer<float, double>[]> m_out_buffer;
  std::vector<std::unique_ptr<float[]>> m_in_model, m_out_model;

  // AUDIO PERFORM
  // bool m_use_thread, m_should_stop_perform_thread;
  // std::unique_ptr<std::thread> m_compute_thread;
  // std::binary_semaphore m_data_available_lock, m_result_available_lock;

  void operator()(audio_bundle input, audio_bundle output);
  void perform(audio_bundle input, audio_bundle output);

  // ONLY FOR DOCUMENTATION
  argument<symbol> path_arg{this, "model path",
                            "Absolute path to the pretrained model."};
  argument<symbol> method_arg{this, "method",
                              "Name of the method to call during synthesis."};
  argument<int> buffer_arg{
      this, "buffer size",
      "Size of the internal buffer (can't be lower than the method's ratio)."};

  // ENABLE / DISABLE ATTRIBUTE
  attribute<bool> enable{this, "enable", true,
                         description{"Enable / disable tensor computation"}};

  // ENABLE / DISABLE ATTRIBUTE
  attribute<bool> gpu{this, "gpu", false,
                      description{"Enable / disable gpu usage when available"},
                      setter{[this](const c74::min::atoms &args,
                                    const int inlet) -> c74::min::atoms {
                        if (m_is_backend_init)
                          m_model->use_gpu(bool(args[0]));
                        return args;
                      }}};

  // Message nn_msg_in
  message<> nn_msg_in {this, "nn_msg_in", "Input Message to model", 
      [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms {

        input_msg.clear();

        // Assert args matches input dimension
        if(args.size() != m_in_dim) {
          cerr << "Input message dims doesn't match model input dims (" << args.size() << " != " << m_in_dim << ")" << endl;
          return {};
        }

        // Assert all args are float|int and set input_msg to args
        int i = 0;
        for(auto a : args) {
          if(a.a_type != c74::max::A_FLOAT && a.a_type != c74::max::A_LONG) {
            cerr << "All values must be floats or ints (" << a << ")" << endl;
            return {};
          }
          if(a.a_type == c74::max::A_FLOAT)
            input_msg.push_back(a.a_w.w_float);
          else
            input_msg.push_back((float)a.a_w.w_long);
          // cout << i << ": " << a.a_w.w_float << " : " << a.a_type << endl;
        }

        return {};
      }
  };

  // BOOT STAMP
  message<> maxclass_setup{
      this, "maxclass_setup",
      [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms {
        cout << "nn~ " << VERSION << " - torch " << TORCH_VERSION
             << " - 2023 - Antoine Caillon & Axel Chemla--Romeu-Santos" << endl;
        cout << "visit https://caillonantoine.github.io" << endl;
        return {};
      }};

  message<> anything{
      this, "anything", "callback for attributes",
      [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms {
        symbol attribute_name = args[0];
        if (attribute_name == "reload") {
          m_model->reload();
        } else if (attribute_name == "get_attributes") {
          for (std::string attr : settable_attributes) {
            cout << attr << endl;
          }
          return {};
        } else if (attribute_name == "get_methods") {
          for (std::string method : m_model->get_available_methods())
            cout << method << endl;
          return {};
        } else if (attribute_name == "get") {
          if (args.size() < 2) {
            cerr << "get must be given an attribute name" << endl;
            return {};
          }
          attribute_name = args[1];
          if (m_model->has_settable_attribute(attribute_name)) {
            cout << attribute_name << ": "
                 << m_model->get_attribute_as_string(attribute_name) << endl;
          } else {
            cerr << "no attribute " << attribute_name << " found in model"
                 << endl;
          }
          return {};
        } else if (attribute_name == "set") {
          if (args.size() < 3) {
            cerr << "set must be given an attribute name and corresponding "
                    "arguments"
                 << endl;
            return {};
          }
          attribute_name = args[1];
          std::vector<std::string> attribute_args;
          if (has_settable_attribute(attribute_name)) {
            for (int i = 2; i < args.size(); i++) {
              attribute_args.push_back(args[i]);
            }
            try {
              m_model->set_attribute(attribute_name, attribute_args);
            } catch (std::string message) {
              cerr << message << endl;
            }
          } else {
            cerr << "model does not have attribute " << attribute_name << endl;
          }
        } else {
          cerr << "no corresponding method for " << attribute_name << endl;
        }
        return {};
      }};
};

nn_msg_tilde::nn_msg_tilde(const atoms &args)
    : m_in_dim(1), m_in_ratio(1), m_out_dim(1),
      m_out_ratio(1), m_buffer_size(4096), m_method("forward") {

  m_model = std::make_unique<Backend_MSG>();
  m_is_backend_init = true;

  // CHECK ARGUMENTS
  if (!args.size()) {
    return;
  }
  if (args.size() > 0) { // ONE ARGUMENT IS GIVEN
    auto model_path = std::string(args[0]);
    if (model_path.substr(model_path.length() - 3) != ".ts")
      model_path = model_path + ".ts";
    m_path = path(model_path);
  }
  if (args.size() > 1) { // TWO ARGUMENTS ARE GIVEN
    m_method = std::string(args[1]);
  }
  if (args.size() > 2) { // THREE ARGUMENTS ARE GIVEN
    m_buffer_size = int(args[2]);
  }

  // TRY TO LOAD MODEL
  if (m_model->load(std::string(m_path))) {
    cerr << "error during loading" << endl;
    error();
    return;
  }

  m_model->use_gpu(gpu);

  m_higher_ratio = m_model->get_higher_ratio();

  // GET MODEL'S METHOD PARAMETERS
  auto params = m_model->get_method_params(m_method);

  // GET MODEL'S SETTABLE ATTRIBUTES
  try {
    settable_attributes = m_model->get_settable_attributes();
  } catch (...) {
  }

  if (!params.size()) {
    error("method " + m_method + " not found !");
  }

  m_in_dim = params[0];
  m_in_ratio = params[1];
  m_out_dim = params[2];
  m_out_ratio = params[3];

  if (!m_buffer_size) {
    // NO THREAD MODE
    m_buffer_size = m_higher_ratio;
  } else if (m_buffer_size < m_higher_ratio) {
    m_buffer_size = m_higher_ratio;
    cerr << "buffer size too small, switching to " << m_buffer_size << endl;
  } else {
    m_buffer_size = power_ceil(m_buffer_size);
  }

  // Calling forward in a thread causes memory leak in windows.
  // See https://github.com/pytorch/pytorch/issues/24237
// #ifdef _WIN32
//   m_use_thread = false;
// #endif

  input_msg.resize(m_in_dim, 0.0);  // Fill input_msg with 0.0's
  
  /**
   *  Create inlets
   *    - For now, 1 inlet
   *    - Implies a single feature vector (msg) as input
   */
  std::string input_label = "";
  for (int i(0); i < m_in_dim; i++)
  {
    try
    {
      input_label += m_model->get_model()
                        .attr(m_method + "_input_labels")
                        .toList()
                        .get(i)
                        .toStringRef() + (i != m_in_dim -1 ? "; " : "");
    }
    catch (...)
    {
      input_label += "(nn_msg) model input " + std::to_string(i) + (i != m_in_dim -1 ? "; " : "");
    }
  }
  m_inlets.push_back(std::make_unique<inlet<>>(this, input_label, "nn_msg_in"));

  // Create outlets
  m_out_buffer = std::make_unique<circular_buffer<float, double>[]>(m_out_dim);
  for (int i(0); i < m_out_dim; i++) {
    std::string output_label = "";
    try {
      output_label = m_model->get_model()
                         .attr(m_method + "_output_labels")
                         .toList()
                         .get(i)
                         .toStringRef();
    } catch (...) {
      output_label = "(signal) model output " + std::to_string(i);
    }
    m_outlets.push_back(
        std::make_unique<outlet<>>(this, output_label, "signal"));
    m_out_buffer[i].initialize(m_buffer_size);
    m_out_model.push_back(std::make_unique<float[]>(m_buffer_size));
  }

  // if (m_use_thread)
  //   m_compute_thread = std::make_unique<std::thread>(model_perform_loop, this);
}

nn_msg_tilde::~nn_msg_tilde() {
  // m_should_stop_perform_thread = true;
  // if (m_compute_thread)
  //   m_compute_thread->join();
}

bool nn_msg_tilde::has_settable_attribute(std::string attribute) {
  for (std::string candidate : settable_attributes) {
    if (candidate == attribute)
      return true;
  }
  return false;
}

void fill_with_zero(audio_bundle output) {
  for (int c(0); c < output.channel_count(); c++) {
    auto out = output.samples(c);
    for (int i(0); i < output.frame_count(); i++) {
      out[i] = 0.;
    }
  }
}

void nn_msg_tilde::operator()(audio_bundle input, audio_bundle output) {
  auto dsp_vec_size = output.frame_count();

  // CHECK IF MODEL IS LOADED AND ENABLED
  if (!m_model->is_loaded() || !enable) {
    fill_with_zero(output);
    return;
  }

  // CHECK IF DSP_VEC_SIZE IS LARGER THAN BUFFER SIZE
  if (dsp_vec_size > m_buffer_size) {
    cerr << "vector size (" << dsp_vec_size << ") ";
    cerr << "larger than buffer size (" << m_buffer_size << "). ";
    cerr << "disabling model.";
    cerr << endl;
    enable = false;
    fill_with_zero(output);
    return;
  }

  perform(input, output);
}

void nn_msg_tilde::perform(audio_bundle input, audio_bundle output) {
  auto vec_size = output.frame_count();

  // Call model
  m_model->perform(input_msg, output_msg, m_method);

  // Add to circular buffer (For now, only a single output channel)
  m_out_buffer[0].put(output_msg.data(), output_msg.size());

  // Copy Circuar Buffer to Output
  for (int c(0); c < output.channel_count(); c++) {
    auto out = output.samples(c);
    m_out_buffer[c].get(out, vec_size);
  }
}

MIN_EXTERNAL(nn_msg_tilde);
