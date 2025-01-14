`timescale 1ns/1ps

module weight_ram #(
   parameter  pWEIGHT_DATA_WIDTH  = 64
  ,parameter  pWEIGHT_BASE_ADDR   = 4000_0000
  
  ,parameter  pKERNEL_NUM         = 1024
  ,parameter  pBLOCK_RAM_NUM      = 8
)(
   input  logic                                         clk
  ,input  logic                                         rst
  ,input  logic                                         wr_en
  ,input  logic [31:0]                                  weight_addr
  ,input  logic [pWEIGHT_DATA_WIDTH-1:0]                weight_data
  ,input  logic [$clog2(pKERNEL_NUM)-1:0]               kernel_addr
  ,output logic [pWEIGHT_DATA_WIDTH*pBLOCK_RAM_NUM-1:0] kernel_data
);
  
  (* ram_style = "ultra" *)
  logic [pWEIGHT_DATA_WIDTH-1:0] weight_r [0:pBLOCK_RAM_NUM-1][0:pKERNEL_NUM-1];
  logic [$clog2(pBLOCK_RAM_NUM)-1:0] ram_idx;
    
  always_ff @(posedge clk) begin
    if (rst || ram_idx == pBLOCK_RAM_NUM-1)
      ram_idx <= 'b0;
    else if (wr_en && pWEIGHT_BASE_ADDR <= weight_addr && weight_addr <= pWEIGHT_BASE_ADDR+pKERNEL_NUM)
      ram_idx <= ram_idx + 1'b1;
  end
  
  genvar idx;

  generate
    for (idx = 0; idx < pBLOCK_RAM_NUM; idx = idx + 1) begin
      always_ff @(posedge clk) begin
        if (wr_en && idx == ram_idx)
          weight_r[idx][weight_addr - pWEIGHT_BASE_ADDR] <= weight_data;
        else
          kernel_data[idx*pWEIGHT_DATA_WIDTH +: pWEIGHT_DATA_WIDTH] <= weight_r[idx][kernel_addr];
      end
    end
  endgenerate

endmodule
