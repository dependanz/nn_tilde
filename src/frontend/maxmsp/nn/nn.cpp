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

class nn_msg : public object<nn_msg>
{
public:
  MIN_DESCRIPTION{"Interface for deep learning models"};
  MIN_TAGS{"audio, deep learning, ai"};
  MIN_AUTHOR{"Antoine Caillon & Axel Chemla--Romeu-Santos"};

  nn_msg(const atoms &args = {});
  ~nn_msg();

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

  // ONLY FOR DOCUMENTATION
  argument<symbol> path_arg{this, "model path",
                            "Absolute path to the pretrained model."};
  argument<symbol> method_arg{this, "method",
                              "Name of the method to call during synthesis."};

  // ENABLE / DISABLE ATTRIBUTE
  attribute<bool> enable{this, "enable", true,
                         description{"Enable / disable tensor computation"}};

  // ENABLE / DISABLE ATTRIBUTE
  attribute<bool> gpu{this, "gpu", false,
                      description{"Enable / disable gpu usage when available"},
                      setter{[this](const c74::min::atoms &args,
                                    const int inlet) -> c74::min::atoms
                             {
                               if (m_is_backend_init)
                                 m_model->use_gpu(bool(args[0]));
                               return args;
                             }}};

  // nn_msg_in messages call the model
  message<> nn_msg_in { this, "nn_msg_in", "Message to input to model",
      [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms
      {
          input_msg.clear();
          // output_msg.clear();

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

          // Call model
          m_model->perform(input_msg, output_msg, m_method);

          // For now, 1 outlet
          c74::min::atoms out_atoms;
          out_atoms.push_back("nn_msg_out");
          for(float o : output_msg) {
            out_atoms.push_back(o);
          }
          m_outlets[0]->send(out_atoms);

          return {};
      }
  };

  // BOOT STAMP
  message<> maxclass_setup{
      this, "maxclass_setup",
      [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms
      {
        cout << "nn " << VERSION << " - torch " << TORCH_VERSION
             << " - 2023 - Antoine Caillon & Axel Chemla--Romeu-Santos" << endl;
        cout << "visit https://caillonantoine.github.io" << endl;
        return {};
      }};

  message<> anything{
      this, "anything", "callback for attributes",
      [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms
      {
        symbol attribute_name = args[0];
        if (attribute_name == "reload")
        {
          m_model->reload();
        }
        else if (attribute_name == "get_attributes")
        {
          for (std::string attr : settable_attributes)
          {
            cout << attr << endl;
          }
          return {};
        }
        else if (attribute_name == "get_methods")
        {
          for (std::string method : m_model->get_available_methods())
            cout << method << endl;
          return {};
        }
        else if (attribute_name == "get")
        {
          if (args.size() < 2)
          {
            cerr << "get must be given an attribute name" << endl;
            return {};
          }
          attribute_name = args[1];
          if (m_model->has_settable_attribute(attribute_name))
          {
            cout << attribute_name << ": "
                 << m_model->get_attribute_as_string(attribute_name) << endl;
          }
          else
          {
            cerr << "no attribute " << attribute_name << " found in model"
                 << endl;
          }
          return {};
        }
        else if (attribute_name == "set")
        {
          if (args.size() < 3)
          {
            cerr << "set must be given an attribute name and corresponding "
                    "arguments"
                 << endl;
            return {};
          }
          attribute_name = args[1];
          std::vector<std::string> attribute_args;
          if (has_settable_attribute(attribute_name))
          {
            for (int i = 2; i < args.size(); i++)
            {
              attribute_args.push_back(args[i]);
            }
            try
            {
              m_model->set_attribute(attribute_name, attribute_args);
            }
            catch (std::string message)
            {
              cerr << message << endl;
            }
          }
          else
          {
            cerr << "model does not have attribute " << attribute_name << endl;
          }
        }
        else
        {
          cerr << "no corresponding method for " << attribute_name << endl;
        }
        return {};
      }};
};

nn_msg::nn_msg(const atoms &args)
    : m_in_dim(1), m_out_dim(1), m_method("forward")
{
  m_model = std::make_unique<Backend_MSG>();
  m_is_backend_init = true;

  // Check Arguments
  if (!args.size())
  {
    return;
  }
  if (args.size() > 0)
  { // 1 Argument => Model Path
    auto model_path = std::string(args[0]);
    if (model_path.substr(model_path.length() - 3) != ".ts")
      model_path = model_path + ".ts";
    m_path = path(model_path);
  }
  if (args.size() > 1)
  { // 2 Arguments => (Model Path, Method Name)
    m_method = std::string(args[1]);
  }

  // Try to Load Model
  if (m_model->load(std::string(m_path)))
  {
    cerr << "error during loading" << endl;
    error();
    return;
  }

  m_model->use_gpu(gpu);

  // Get Model's Method Params
  auto params = m_model->get_method_params(m_method);

  // Get Model's Settable Attributes
  try
  {
    settable_attributes = m_model->get_settable_attributes();
  }
  catch (...)
  {
  }

  if (!params.size())
  {
    error("method " + m_method + " not found !");
  }

  /**
   * Keeping the format of the params the same, in case
   * one wishes to use nn~ with the same model.
   */
  m_in_dim = params[0];
  m_in_ratio = params[1];
  m_out_dim = params[2];
  m_out_ratio = params[3];

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
  
  /**
   *  Create Outlets
   *    - For now, 1 outlet.
   *    - Implies a mono-channel frame (msg) as output
   */
  std::string output_label = "";
  for (int i(0); i < m_out_dim; i++)
  {
    std::string output_label = "";
    try
    {
      output_label += m_model->get_model()
                         .attr(m_method + "_output_labels")
                         .toList()
                         .get(i)
                         .toStringRef() + (i != m_in_dim -1 ? "; " : "");
    }
    catch (...)
    {
      output_label += "(nn_msg) model output " + std::to_string(i) + (i != m_in_dim -1 ? "; " : "");
    }
  }
  m_outlets.push_back(std::make_unique<outlet<>>(this, output_label, "nn_msg_out"));
}

nn_msg::~nn_msg()
{
}

bool nn_msg::has_settable_attribute(std::string attribute)
{
  for (std::string candidate : settable_attributes)
  {
    if (candidate == attribute)
      return true;
  }
  return false;
}

MIN_EXTERNAL(nn_msg);
