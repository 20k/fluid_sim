#ifndef UTIL_HPP_INCLUDED
#define UTIL_HPP_INCLUDED

#include <iomanip>
#include <math.h>
#include <SFML/Graphics.hpp>

template <typename T>
inline
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out << std::setprecision(n) << a_value;
    return out.str();
}

template<typename T>
inline
std::string to_string_with_variable_prec(const T a_value)
{
    int n = ceil(log10(a_value)) + 1;

    if(a_value <= 0.0001f)
        n = 2;

    if(n < 2)
        n = 2;

    std::ostringstream out;
    out << std::setprecision(n) << a_value;
    return out.str();
}

template<typename T>
inline
std::string to_string_with_enforced_variable_dp(T a_value, int forced_dp = 1)
{
    if(fabs(a_value) <= 0.0999999 && fabs(a_value) >= 0.0001)
        forced_dp++;

    /*a_value = a_value * pow(10, forced_dp + 1);
    a_value = round(a_value);
    a_value = a_value * pow(10, -forced_dp - 1);*/

    std::string fstr = std::to_string(a_value);

    auto found = fstr.find('.');

    if(found == std::string::npos)
    {
        return fstr + ".0";
    }

    found += forced_dp + 1;

    if(found >= fstr.size())
        return fstr;

    fstr.resize(found);

    return fstr;
}

inline
std::string format(std::string to_format, const std::vector<std::string>& all_strings)
{
    int len = 0;

    for(auto& i : all_strings)
    {
        if(i.length() > len)
            len = i.length();
    }

    for(int i=to_format.length(); i<len; i++)
    {
        to_format = to_format + " ";
    }

    return to_format;
}


template<sf::Keyboard::Key k, int n, int c>
bool once()
{
    static bool last;

    sf::Keyboard key;

    if(key.isKeyPressed(k) && !last)
    {
        last = true;

        return true;
    }

    if(!key.isKeyPressed(k))
    {
        last = false;
    }

    return false;
}

template<sf::Mouse::Button b, int n, int c>
bool once()
{
    static bool last;

    sf::Mouse mouse;

    if(mouse.isButtonPressed(b) && !last)
    {
        last = true;

        return true;
    }

    if(!mouse.isButtonPressed(b))
    {
        last = false;
    }

    return false;
}

#define ONCE_MACRO(x) once<x, __LINE__, __COUNTER__>()

inline
bool key_down(sf::Keyboard::Key k)
{
    sf::Keyboard key;

    return key.isKeyPressed(k);
}

inline
std::string obfuscate(const std::string& str, bool should_obfuscate)
{
    if(!should_obfuscate)
        return str;

    std::string ret = str;

    for(int i=0; i<ret.length(); i++)
    {
        if(isalnum(ret[i]))
        {
            ret[i] = '?';
        }
    }

    return ret;
}

#endif // UTIL_HPP_INCLUDED
