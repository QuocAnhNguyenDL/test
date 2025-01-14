`timescale 1ns/1ps

module pe_pooling_datapath #(
   parameter  pDATA_WIDTH   = 8
  ,parameter  pWINDOW_SIZE  = 3
  
  ,parameter  pPOOLING_TYPE = "max"
)( 
   input  logic                                 clk
  ,input  logic                                 rst
  ,input  logic                                 en
  ,input  logic [pDATA_WIDTH*pWINDOW_SIZE-1:0]  data_in
  ,output logic [pDATA_WIDTH-1:0]               data_out
);

  localparam pPOOLING_STAGE_NUM = $clog2(pWINDOW_SIZE) + 1;
   
  logic [pDATA_WIDTH-1:0] pooling_out_r [0:pPOOLING_STAGE_NUM-1][0:pWINDOW_SIZE-1];
  logic [pPOOLING_STAGE_NUM-1:0] valid_r;

  genvar stage_idx;
  genvar reg_idx;
  
  generate
    for (stage_idx = 0; stage_idx < pPOOLING_STAGE_NUM; stage_idx = stage_idx+1) begin
      localparam pPRE_STAGE_REG_NUM = int'($ceil(real'(pWINDOW_SIZE)/real'(2**(stage_idx-1))));    
      localparam pCURR_STAGE_REG_NUM = int'($ceil(real'(pWINDOW_SIZE)/real'(2**stage_idx)));
      
      logic valid_in;
            
      assign valid_in = stage_idx ? valid_r[stage_idx-1] : en;
    
      always_ff @(posedge clk) begin
        if (rst)
          valid_r[stage_idx] <= 'b0;
        else
          valid_r[stage_idx] <= valid_in;
      end       
                  
      for (reg_idx = 0; reg_idx < pCURR_STAGE_REG_NUM; reg_idx = reg_idx+1) begin
        logic [pDATA_WIDTH-1:0] pooling_in;
      
        if (stage_idx == 0)
          assign pooling_in = data_in[reg_idx*pDATA_WIDTH +: pDATA_WIDTH];
        else
          if (reg_idx*2 == pPRE_STAGE_REG_NUM-1)
            assign pooling_in = pooling_out_r[stage_idx-1][reg_idx*2];
          else
            if (pPOOLING_TYPE == "max")
              assign pooling_in = pooling_out_r[stage_idx-1][reg_idx*2] > pooling_out_r[stage_idx-1][reg_idx*2+1] ? pooling_out_r[stage_idx-1][reg_idx*2] : pooling_out_r[stage_idx-1][reg_idx*2+1];
            else if (pPOOLING_TYPE == "min")
              assign pooling_in = pooling_out_r[stage_idx-1][reg_idx*2] < pooling_out_r[stage_idx-1][reg_idx*2+1] ? pooling_out_r[stage_idx-1][reg_idx*2] : pooling_out_r[stage_idx-1][reg_idx*2+1];
            else if (pPOOLING_TYPE == "average")                       
              assign pooling_in = pooling_out_r[stage_idx-1][reg_idx*2] + pooling_out_r[stage_idx-1][reg_idx*2+1];
            else
              $fatal(1, "Invalid Configuration detected - aborting");
          
        always_ff @(posedge clk) begin
          if (rst)
            pooling_out_r[stage_idx][reg_idx] <= 'b0;
          else if (valid_in)
            pooling_out_r[stage_idx][reg_idx] <= pooling_in;
        end
      end
    end
  endgenerate

  assign data_out = pooling_out_r[pPOOLING_STAGE_NUM-1][0];

endmodule

