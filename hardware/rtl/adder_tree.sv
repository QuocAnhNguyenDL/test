`timescale 1ns/1ps

module adder_tree #(
   parameter  pDATA_WIDTH     = 32
  ,parameter  pINPUT_NUM      = 32
)( 
   input  logic                               clk
  ,input  logic                               rst
  ,input  logic                               en
  ,input  logic [pDATA_WIDTH*pINPUT_NUM-1:0]  data_in
  ,output logic [pDATA_WIDTH-1:0]             data_out
);

  localparam pADDER_STAGE_NUM = $clog2(pINPUT_NUM) + 1;
   
  logic signed [pDATA_WIDTH-1:0] adder_out_r [0:pADDER_STAGE_NUM-1][0:pINPUT_NUM-1];
  logic [pADDER_STAGE_NUM-1:0] valid_r;

  genvar stage_idx;
  genvar reg_idx;
  
  generate
    for (stage_idx = 0; stage_idx < pADDER_STAGE_NUM; stage_idx = stage_idx+1) begin
      localparam pPRE_STAGE_REG_NUM = int'($ceil(real'(pINPUT_NUM)/real'(2**(stage_idx-1))));    
      localparam pCURR_STAGE_REG_NUM = int'($ceil(real'(pINPUT_NUM)/real'(2**stage_idx)));
      
      logic valid_in;
      
      if (stage_idx)
        assign valid_in = valid_r[stage_idx-1];
      else
        assign valid_in = en;
    
      always_ff @(posedge clk) begin
        if (rst)
          valid_r[stage_idx] <= 1'b0;
        else if (valid_in)
          valid_r[stage_idx] <= 1'b1;
      end       
                  
      for (reg_idx = 0; reg_idx < pCURR_STAGE_REG_NUM; reg_idx = reg_idx+1) begin
        logic signed [pDATA_WIDTH-1:0] adder_in;
      
        if (stage_idx == 0)
          assign adder_in = data_in[reg_idx*pDATA_WIDTH +: pDATA_WIDTH];
        else
          if (reg_idx*2 == pPRE_STAGE_REG_NUM-1)
            assign adder_in = adder_out_r[stage_idx-1][reg_idx*2];
          else
            assign adder_in = adder_out_r[stage_idx-1][reg_idx*2] + adder_out_r[stage_idx-1][reg_idx*2+1];
          
        always_ff @(posedge clk) begin
          if (rst)
            adder_out_r[stage_idx][reg_idx] <= 'b0;
          else if (valid_in)
            adder_out_r[stage_idx][reg_idx] <= adder_in;
        end
      end
    end
  endgenerate

  assign data_out = adder_out_r[pADDER_STAGE_NUM-1][0];

endmodule
