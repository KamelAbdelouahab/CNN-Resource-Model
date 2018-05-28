library ieee;
    use	ieee.std_logic_1164.all;
    use	ieee.numeric_std.all;

entity constMult is
generic (
KERNEL : std_logic_vector(7 downto 0) := std_logic_vector(to_signed(126,8))
    );
port(
    d      : in  std_logic_vector(7  downto 0);
    o      : out std_logic_vector(15 downto 0)
    );
end entity;

architecture rtl of constMult is
begin
    o <= std_logic_vector(signed(d) * signed(KERNEL));
end rtl;
