#include "conversion.hpp"

#include <string>
#include <vector>
#include <stdexcept>

#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/core/memory.hpp>

namespace godot
{

    std::vector<float> float32_array_to_vec(PackedFloat32Array array)
    {
        std::vector<float> vec{};
        for (float f : array)
        {
            vec.push_back(f);
        }
        return vec;
    }

    PackedFloat32Array float32_vec_to_array(std::vector<float> vec)
    {
        PackedFloat32Array array{};
        for (float f : vec)
        {
            array.push_back(f);
        }
        return array;
    }

    std::vector<int> gd_arr_to_int_vec(Array arr)
    {
        std::vector<int> vec;
        for (int i = 0; i < arr.size(); i++)
        {
            Variant item = arr[i];
            auto item_type = item.get_type();

            if (item_type != Variant::INT)
                throw std::runtime_error("An item in the array is not an int");

            vec.emplace_back((int)item);
        }

        return vec;
    }

    Array int_vec_to_gd_arr(std::vector<int> vec)
    {
        Array arr = Array();
        for (const int &item : vec)
        {
            arr.append(item);
        }

        return arr;
    }

    std::string string_gd_to_std(String s)
    {
        std::string new_s{s.utf8().get_data()};
        return new_s;
    }

    String string_std_to_gd(std::string s)
    {
        String new_s;
        new_s.parse_utf8(s.data());
        return new_s;
    }

    // https://stackoverflow.com/questions/28270310/how-to-easily-detect-utf8-encoding-in-the-string
    bool is_utf8(const char *string)
    {
        if (!string)
            return true;

        const unsigned char *bytes = (const unsigned char *)string;
        int num;

        while (*bytes != 0x00)
        {
            if ((*bytes & 0x80) == 0x00)
            {
                // U+0000 to U+007F
                num = 1;
            }
            else if ((*bytes & 0xE0) == 0xC0)
            {
                // U+0080 to U+07FF
                num = 2;
            }
            else if ((*bytes & 0xF0) == 0xE0)
            {
                // U+0800 to U+FFFF
                num = 3;
            }
            else if ((*bytes & 0xF8) == 0xF0)
            {
                // U+10000 to U+10FFFF
                num = 4;
            }
            else
                return false;

            bytes += 1;
            for (int i = 1; i < num; ++i)
            {
                if ((*bytes & 0xC0) != 0x80)
                    return false;
                bytes += 1;
            }
        }

        return true;
    }

}
