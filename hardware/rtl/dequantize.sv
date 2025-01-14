`timescale 1ns/1ps

(* use_dsp = "yes", KEEP = "true" *)
module dequantize (
   input  logic                 clk
  ,input  logic                 en
  ,input  logic                 rst
  ,input  logic signed  [31:0]  data_in
  ,input  logic signed  [31:0]  scale
//  ,input  logic signed  [17:0]  data_in
//  ,input  logic signed  [26:0]  scale
  ,output logic signed  [31:0]  data_out
);

  logic [63:0] mult_out;
    
  always_ff @(posedge clk) begin
    if (rst)
      mult_out <= 'b0;
    else if (en)
      mult_out <= data_in * scale;
      //mult_out <= data_in;
  end
  
  assign data_out = mult_out[47:16];
  
endmodule
