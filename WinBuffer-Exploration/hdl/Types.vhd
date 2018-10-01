library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.math_real.all;

package types is
  ------------------------------------------------------------------------------
  -- Types
  constant GENERAL_BITWIDTH      : integer := 6;
  type pixel_array is array (integer range <>) of std_logic_vector (GENERAL_BITWIDTH-1 downto 0);
end types;
