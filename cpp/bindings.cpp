#include "tokenizer.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

py::bytes decode_token_ids(const BPETokenizer &tokenizer, const py::object &token_ids)
{
    if (py::isinstance<py::array>(token_ids))
    {
        auto array = py::array_t<std::uint32_t, py::array::c_style | py::array::forcecast>::ensure(token_ids);
        if (!array || array.ndim() != 1)
            throw std::invalid_argument("token IDs must be a one-dimensional array");

        return py::bytes(tokenizer.decode_bytes(std::span(array.data(), array.size())));
    }

    return py::bytes(tokenizer.decode_bytes(py::cast<std::vector<int>>(token_ids)));
}

py::dict state_to_dict(const TokenizerState &state)
{
    py::list merges;

    for (const MergeRule &rule : state.merges)
    {
        py::list merge;
        merge.append(rule.left_token);
        merge.append(rule.right_token);
        merge.append(rule.new_token);
        merges.append(merge);
    }

    py::dict result;
    result["format"] = state.format;
    result["language"] = state.language;
    result["pre_tokenizer"] = state.pre_tokenizer;
    result["mergeable_vocab_size"] = state.mergeable_vocab_size;
    result["vocab_size"] = state.vocab_size;
    result["special_tokens"] = state.special_tokens;
    result["regex"] = state.regex;
    result["merges"] = merges;

    return result;
}

TokenizerState dict_to_state(const py::dict &data)
{
    TokenizerState state;
    state.format = py::cast<std::string>(data["format"]);
    state.language = py::cast<std::string>(data["language"]);
    state.pre_tokenizer = py::cast<std::string>(data["pre_tokenizer"]);
    state.mergeable_vocab_size = py::cast<int>(data["mergeable_vocab_size"]);
    state.vocab_size = py::cast<int>(data["vocab_size"]);
    state.special_tokens = py::cast<std::map<std::string, int>>(data["special_tokens"]);
    state.regex = py::cast<std::string>(data["regex"]);

    std::vector<std::vector<int>> merges = py::cast<std::vector<std::vector<int>>>(data["merges"]);

    for (const std::vector<int> &rule : merges)
    {
        if (rule.size() != 3)
            throw std::invalid_argument("each merge rule must contain three token IDs");

        state.merges.push_back({rule[0], rule[1], rule[2]});
    }

    return state;
}

PYBIND11_MODULE(_tokenizer_cpp, module)
{
    py::class_<BPETokenizer>(module, "BPETokenizer")
        .def(py::init<int, const std::map<std::string, int> &>(), py::arg("mergeable_vocab_size"), py::arg("special_tokens") = std::map<std::string, int>{})
        .def("train", &BPETokenizer::train)
        .def("encode", &BPETokenizer::encode, py::arg("text"), py::arg("allowed_special") = std::set<std::string>{})
        .def("decode_bytes", [](const BPETokenizer &tokenizer, const py::object &token_ids) {
            return decode_token_ids(tokenizer, token_ids);
        })
        .def("decode", [](const BPETokenizer &tokenizer, const py::object &token_ids, const std::string &errors) {
            py::bytes decoded = decode_token_ids(tokenizer, token_ids);
            return decoded.attr("decode")("utf-8", errors);
        }, py::arg("token_ids"), py::arg("errors") = "replace")
        .def("to_dict", [](const BPETokenizer &tokenizer) {
            return state_to_dict(tokenizer.to_state());
        })
        .def_static("from_dict", [](const py::dict &state) {
            return BPETokenizer::from_state(dict_to_state(state));
        })
        .def_property_readonly("mergeable_vocab_size", &BPETokenizer::get_mergeable_vocab_size)
        .def_property_readonly("vocab_size", &BPETokenizer::get_vocab_size)
        .def_property_readonly("merges", &BPETokenizer::get_merges)
        .def_property_readonly("special_tokens", &BPETokenizer::get_special_tokens);
}
