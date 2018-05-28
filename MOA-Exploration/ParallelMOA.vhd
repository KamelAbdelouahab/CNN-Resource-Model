library ieee;
    use ieee.std_logic_1164.all;
    use ieee.std_logic_signed.all;

library work;
    use work.DataTypes.all;

entity ParallelMOA is
    generic(
    DATA_WIDTH   : natural := CONST_DATA_WIDTH;
    NUM_OPERANDS : natural := CONST_NUM_OPERANDS;
    SUM_WIDTH    : natural := CONST_SUM_WIDTH
    );
    port(
    clk_dsp    : in  std_logic;
    clk_sys    : in  std_logic;
    reset_n    : in  std_logic;
    in_data    : in  data_array (0 to NUM_OPERANDS-1);
    in_valid   : in  std_logic;
    out_data   : out std_logic_vector (SUM_WIDTH-1 downto 0)
    );
end entity ParallelMOA;

architecture Bhv of ParallelMOA is
-----------------------------
-- SIGNALS
-----------------------------
signal s_acc     : std_logic_vector (SUM_WIDTH-1 downto 0) := (others=>'0');

begin
  -- Implementation of a Multi Operand Adder As an Adder trees
    acc_process : process(clk_sys)
    variable v_acc : std_logic_vector (SUM_WIDTH-1 downto 0) := (others=>'0');
    begin
        if (rising_edge(clk_sys)) then
            acc_loop : for i in 0 to NUM_OPERANDS-1 loop
                v_acc := v_acc + in_data(i);
            end loop acc_loop;

        end if;
    s_acc <= v_acc;
    end process;
    out_data <= s_acc;
end architecture Bhv;
