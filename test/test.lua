local nn = require 'nn'

local mytest = torch.TestSuite()
local tester = torch.Tester()

local weldon = require 'WeldonPooling'
local gmp = require 'GlobalMaxPooling'
local gap = require 'GlobalAveragePooling'
local gkmp = require 'GlobalkMaxPooling'
local lse = require 'LogSumExpPooling'

function mytest.testWeldonPooling()
    tester:assert(weldon)
end

function mytest.testGlobalMaxPooling()
    tester:assert(gmp)
end

function mytest.testGlobalAveragePooling()
    tester:assert(gap)
end

function mytest.testGlobalkMaxPooling()
    tester:assert(gkmp)
end

function mytest.testLogSumExpPooling()
    tester:assert(lse)
end

function mytest.testWeldonPoolingForwardBatch()
    local m = nn.WeldonPooling()
    local input = torch.ones(10, 5, 10, 10)
    local output = m:forward(input)
    tester:eq(output, (torch.ones(10, 5, 1, 1)):mul(2))

    m = nn.WeldonPooling(5)
    output = m:forward(input)
    tester:eq(output, (torch.ones(10, 5, 1, 1)):mul(2))

    m = nn.WeldonPooling(5, 1)
    output = m:forward(input)
    tester:eq(output, (torch.ones(10, 5, 1, 1)):mul(2))

    m = nn.WeldonPooling(1, 0)
    output = m:forward(input)
    tester:eq(output, (torch.ones(10, 5, 1, 1)))

    m = nn.WeldonPooling(5, 0)
    output = m:forward(input)
    tester:eq(output, (torch.ones(10, 5, 1, 1)))

    input:mul(2)

    m = nn.WeldonPooling(2, 2)
    output = m:forward(input)
    tester:eq(output, (torch.ones(10, 5, 1, 1)):mul(4))

    input = torch.zeros(2, 2, 3, 3)
    input[1][1][1][1] = 2
    input[1][1][1][2] = 2
    input[1][1][3][3] = -1
    input[1][1][2][3] = -1

    local outputTrue = torch.zeros(2, 2, 1, 1)
    outputTrue[1][1][1][1] = 1
    output = m:forward(input)
    tester:eq(output, outputTrue)

    m = nn.WeldonPooling(1/9, 1/9)
    output = m:forward(input)
    tester:eq(output, outputTrue)

    m = nn.WeldonPooling(2/9, 1/9)
    output = m:forward(input)
    tester:eq(output, outputTrue)

    m = nn.WeldonPooling(3/9, 1/9)
    output = m:forward(input)
    outputTrue[1][1][1][1] = 1/3
    tester:eq(output, outputTrue, 1e-15)

    m = nn.WeldonPooling(3/9, 3/9)
    output = m:forward(input)
    outputTrue[1][1][1][1] = 2/3
    tester:eq(output, outputTrue)

    m = nn.WeldonPooling(5/9, 5/9)
    output = m:forward(input)
    outputTrue[1][1][1][1] = 2/5
    tester:eq(output, outputTrue)

end

function mytest.testWeldonPoolingForwardSingle()
    local m = nn.WeldonPooling()
    local input = torch.ones(5, 10, 10)
    local output = m:forward(input)
    tester:eq(output, (torch.ones(5, 1, 1)):mul(2))

    m = nn.WeldonPooling(5)
    output = m:forward(input)
    tester:eq(output, (torch.ones(5, 1, 1)):mul(2))

    m = nn.WeldonPooling(5, 1)
    output = m:forward(input)
    tester:eq(output, (torch.ones(5, 1, 1)):mul(2))

    m = nn.WeldonPooling(1, 0)
    output = m:forward(input)
    tester:eq(output, (torch.ones(5, 1, 1)))

    m = nn.WeldonPooling(5, 0)
    output = m:forward(input)
    tester:eq(output, (torch.ones(5, 1, 1)))

    input:mul(2)

    m = nn.WeldonPooling(2, 2)
    output = m:forward(input)
    tester:eq(output, (torch.ones(5, 1, 1)):mul(4))

    input = torch.zeros(2, 3, 3)
    input[1][1][1] = 2
    input[1][1][2] = 2
    input[1][3][3] = -1
    input[1][2][3] = -1

    local outputTrue = torch.zeros(2, 1, 1)
    outputTrue[1][1][1] = 1
    output = m:forward(input)
    tester:eq(output, outputTrue)

    m = nn.WeldonPooling(1/9, 1/9)
    output = m:forward(input)
    tester:eq(output, outputTrue)

    m = nn.WeldonPooling(2/9, 1/9)
    output = m:forward(input)
    tester:eq(output, outputTrue)

    m = nn.WeldonPooling(3/9, 1/9)
    output = m:forward(input)
    outputTrue[1][1][1] = 1/3
    tester:eq(output, outputTrue, 1e-15)

    m = nn.WeldonPooling(3/9, 3/9)
    output = m:forward(input)
    outputTrue[1][1][1] = 2/3
    tester:eq(output, outputTrue)

    m = nn.WeldonPooling(5/9, 5/9)
    output = m:forward(input)
    outputTrue[1][1][1] = 2/5
    tester:eq(output, outputTrue)


end

function mytest.testWeldonPoolingBackwardBatch()

    local m = nn.WeldonPooling(2, 2)
    input = torch.zeros(2, 2, 3, 3)
    input[1][1][1][1] = 2
    input[1][1][1][2] = 1
    input[1][1][1][3] = 0.5
    input[1][1][3][3] = -2
    input[1][1][2][3] = -1

    local gradOutput = torch.zeros(2, 2, 1, 1)

    local output = m:forward(input)
    local gradInput = m:backward(input, gradOutput)
    tester:eq(gradInput, torch.zeros(2, 2, 3, 3))

    gradOutput[1][1][1][1] = 2
    gradInput = m:backward(input, gradOutput)

    local gradInputTrue = torch.zeros(2, 2, 3, 3)
    gradInputTrue[1][1][1][1] = 1
    gradInputTrue[1][1][1][2] = 1
    gradInputTrue[1][1][3][3] = 1
    gradInputTrue[1][1][2][3] = 1

    tester:eq(gradInput, gradInputTrue)

    m = nn.WeldonPooling(2/9, 2/9)
    output = m:forward(input)
    gradInput = m:backward(input, gradOutput)
    tester:eq(gradInput, gradInputTrue)

    m = nn.WeldonPooling(1/9, 2/9)
    output = m:forward(input)
    gradInput = m:backward(input, gradOutput)

    gradInputTrue[1][1][1][1] = 2
    gradInputTrue[1][1][1][2] = 0

    tester:eq(gradInput, gradInputTrue)

    m = nn.WeldonPooling(3/9, 0)
    output = m:forward(input)
    gradInput = m:backward(input, gradOutput)

    gradInputTrue[1][1][1][1] = 2/3
    gradInputTrue[1][1][1][2] = 2/3
    gradInputTrue[1][1][1][3] = 2/3
    gradInputTrue[1][1][3][3] = 0
    gradInputTrue[1][1][2][3] = 0

    tester:eq(gradInput, gradInputTrue)
end

function mytest.testWeldonPoolingBackwardSingle()

    local m = nn.WeldonPooling(2, 2)
    input = torch.zeros(2, 3, 3)
    input[1][1][1] = 2
    input[1][1][2] = 2
    input[1][3][3] = -1
    input[1][2][3] = -1

    local gradOutput = torch.zeros(2, 1, 1)

    local output = m:forward(input)
    local gradInput = m:backward(input, gradOutput)
    tester:eq(gradInput, torch.zeros(2, 3, 3))

    gradOutput[1][1][1] = 2
    gradInput = m:backward(input, gradOutput)

    local gradInputTrue = torch.zeros(2, 3, 3)
    gradInputTrue[1][1][1] = 1
    gradInputTrue[1][1][2] = 1
    gradInputTrue[1][3][3] = 1
    gradInputTrue[1][2][3] = 1

    tester:eq(gradInput, gradInputTrue)

    m = nn.WeldonPooling(2/9, 2/9)
    output = m:forward(input)
    gradInput = m:backward(input, gradOutput)
    tester:eq(gradInput, gradInputTrue)

    m = nn.WeldonPooling(1/9, 2/9)
    output = m:forward(input)
    gradInput = m:backward(input, gradOutput)

    gradInputTrue[1][1][1] = 2
    gradInputTrue[1][1][2] = 0

    tester:eq(gradInput, gradInputTrue)

    m = nn.WeldonPooling(3/9, 0)
    output = m:forward(input)
    gradInput = m:backward(input, gradOutput)

    gradInputTrue[1][1][1] = 2/3
    gradInputTrue[1][1][2] = 2/3
    gradInputTrue[1][1][3] = 2/3
    gradInputTrue[1][3][3] = 0
    gradInputTrue[1][2][3] = 0

    tester:eq(gradInput, gradInputTrue)

end

-------------------------------------------------------------------------------
-- Test GlobalMaxPooling
-------------------------------------------------------------------------------

function mytest.testGlobalMaxPoolingForwardBatch()

    local input = torch.zeros(2, 2, 3, 3)
    input[1][1][1][1] = 2
    input[2][2][3][3] = 3

    local outputTrue = torch.zeros(2, 2, 1, 1)
    outputTrue[1][1][1][1] = 2
    outputTrue[2][2][1][1] = 3

    local m = nn.GlobalMaxPooling()
    local output = m:forward(input)
    tester:eq(output, outputTrue)

end

function mytest.testGlobalMaxPoolingForwardSingle()

    local input = torch.zeros(2, 3, 3)
    input[1][1][1] = 2
    input[2][3][3] = 3

    local outputTrue = torch.zeros(2, 1, 1)
    outputTrue[1][1][1] = 2
    outputTrue[2][1][1] = 3

    local m = nn.GlobalMaxPooling()
    local output = m:forward(input)
    tester:eq(output, outputTrue)

end

function mytest.testGlobalMaxPoolingBackwardBatch()

    local input = torch.zeros(2, 2, 3, 3)
    input[1][1][1][1] = 2
    input[2][2][3][3] = 3

    local m = nn.GlobalMaxPooling()
    local gradOutput = torch.zeros(2, 2, 1, 1)

    local output = m:forward(input)
    local gradInput = m:backward(input, gradOutput)
    tester:eq(gradInput, torch.zeros(2, 2, 3, 3))

    gradOutput[1][1][1][1] = 2
    gradOutput[2][2][1][1] = 3
    gradInput = m:backward(input, gradOutput)

    local gradInputTrue = torch.zeros(2, 2, 3, 3)
    gradInputTrue[1][1][1][1] = 2
    gradInputTrue[2][2][3][3] = 3
    tester:eq(gradInput, gradInputTrue)

end

function mytest.testGlobalMaxPoolingBackwardSingle()

    local input = torch.zeros(2, 3, 3)
    input[1][1][1] = 2
    input[2][3][3] = 3

    local m = nn.GlobalMaxPooling()
    local gradOutput = torch.zeros(2, 1, 1)

    local output = m:forward(input)
    local gradInput = m:backward(input, gradOutput)
    tester:eq(gradInput, torch.zeros(2, 3, 3))

    gradOutput[1][1][1] = 2
    gradOutput[2][1][1] = 3
    gradInput = m:backward(input, gradOutput)

    local gradInputTrue = torch.zeros(2, 3, 3)
    gradInputTrue[1][1][1] = 2
    gradInputTrue[2][3][3] = 3
    tester:eq(gradInput, gradInputTrue)

end

-------------------------------------------------------------------------------
-- Test GlobalAveragePooling
-------------------------------------------------------------------------------

function mytest.testGlobalAveragePoolingForwardBatch()

    local input = torch.zeros(2, 2, 3, 3)
    input[1][1][1][1] = 18
    input[2][2][3][3] = 9

    local outputTrue = torch.zeros(2, 2, 1, 1)
    outputTrue[1][1][1][1] = 2
    outputTrue[2][2][1][1] = 1

    local m = nn.GlobalAveragePooling()
    local output = m:forward(input)
    tester:eq(output, outputTrue)

end

function mytest.testGlobalAveragePoolingForwardSingle()

    local input = torch.zeros(2, 3, 3)
    input[1][1][1] = 18
    input[2][3][3] = 9

    local outputTrue = torch.zeros(2, 1, 1)
    outputTrue[1][1][1] = 2
    outputTrue[2][1][1] = 1

    local m = nn.GlobalAveragePooling()
    local output = m:forward(input)
    tester:eq(output, outputTrue)

end

function mytest.testGlobalAveragePoolingBackwardBatch()

    local input = torch.ones(2, 2, 3, 3)

    local m = nn.GlobalAveragePooling()
    local gradOutput = torch.zeros(2, 2, 1, 1)

    local output = m:forward(input)
    local gradInput = m:backward(input, gradOutput)
    tester:eq(gradInput, torch.zeros(2, 2, 3, 3))

    local gradOutput = (torch.ones(2, 2, 1, 1)):mul(9)
    gradInput = m:backward(input, gradOutput)

    local gradInputTrue = torch.ones(2, 2, 3, 3)
    tester:eq(gradInput, gradInputTrue)

end

function mytest.testGlobalAveragePoolingBackwardSingle()

    local input = torch.ones(2, 3, 3)

    local m = nn.GlobalAveragePooling()
    local gradOutput = torch.zeros(2, 1, 1)

    local output = m:forward(input)
    local gradInput = m:backward(input, gradOutput)
    tester:eq(gradInput, torch.zeros(2, 3, 3))

    local gradOutput = (torch.ones(2, 1, 1)):mul(9)
    gradInput = m:backward(input, gradOutput)

    local gradInputTrue = torch.ones(2, 3, 3)
    tester:eq(gradInput, gradInputTrue)

end

-------------------------------------------------------------------------------
-- Test GlobalkMaxPooling
-------------------------------------------------------------------------------

function mytest.testGlobalkMaxPoolingForwardBatch()

    local input = torch.zeros(2, 2, 3, 3)
    input[1][1][1][1] = 4
    input[1][1][1][2] = 2
    input[2][2][3][3] = 2

    local outputTrue = torch.zeros(2, 2, 1, 1)
    outputTrue[1][1][1][1] = 4
    outputTrue[2][2][1][1] = 2

    local m = nn.GlobalkMaxPooling()
    local output = m:forward(input)
    tester:eq(output, outputTrue)

    m = nn.GlobalkMaxPooling(1/9)
    output = m:forward(input)
    tester:eq(output, outputTrue)

    outputTrue = torch.zeros(2, 2, 1, 1)
    outputTrue[1][1][1][1] = 3
    outputTrue[2][2][1][1] = 1

    m = nn.GlobalkMaxPooling(2)
    output = m:forward(input)
    tester:eq(output, outputTrue)

    m = nn.GlobalkMaxPooling(2/9)
    output = m:forward(input)
    tester:eq(output, outputTrue)

    input = torch.ones(2, 2, 3, 3)
    m = nn.GlobalkMaxPooling(-1)
    output = m:forward(input)
    tester:eq(output, torch.ones(2, 2, 1, 1))

end

function mytest.testGlobalkMaxPoolingForwardSingle()

    local input = torch.zeros(2, 3, 3)
    input[1][1][1] = 4
    input[1][1][2] = 2
    input[2][3][3] = 2

    local outputTrue = torch.zeros(2, 1, 1)
    outputTrue[1][1][1] = 4
    outputTrue[2][1][1] = 2

    local m = nn.GlobalkMaxPooling()
    local output = m:forward(input)
    tester:eq(output, outputTrue)

    m = nn.GlobalkMaxPooling(1/9)
    output = m:forward(input)
    tester:eq(output, outputTrue)

    m = nn.GlobalkMaxPooling(0.000001)
    output = m:forward(input)
    tester:eq(output, outputTrue)

    outputTrue = torch.zeros(2, 1, 1)
    outputTrue[1][1][1] = 3
    outputTrue[2][1][1] = 1

    m = nn.GlobalkMaxPooling(2)
    output = m:forward(input)
    tester:eq(output, outputTrue)

    m = nn.GlobalkMaxPooling(2/9)
    output = m:forward(input)
    tester:eq(output, outputTrue)

    input = torch.ones(2, 3, 3)
    m = nn.GlobalkMaxPooling(-1)
    output = m:forward(input)
    tester:eq(output, torch.ones(2, 1, 1))

    m = nn.GlobalkMaxPooling(0.999999)
    output = m:forward(input)
    tester:eq(output, torch.ones(2, 1, 1))

end

function mytest.testGlobalkMaxPoolingBackwardBatch()

    local input = torch.zeros(2, 2, 3, 3)
    input[1][1][1][1] = 4
    input[1][1][1][2] = 2
    input[2][2][3][3] = 2
    input[2][2][3][2] = 1

    local m = nn.GlobalkMaxPooling()
    local gradOutput = torch.zeros(2, 2, 1, 1)

    local output = m:forward(input)
    local gradInput = m:backward(input, gradOutput)
    tester:eq(gradInput, torch.zeros(2, 2, 3, 3))

    gradOutput[1][1][1][1] = 2
    gradOutput[2][2][1][1] = 3
    gradInput = m:backward(input, gradOutput)

    gradInputTrue = torch.zeros(2, 2, 3, 3)
    gradInputTrue[1][1][1][1] = 2
    gradInputTrue[2][2][3][3] = 3
    tester:eq(gradInput, gradInputTrue)

    local m = nn.GlobalkMaxPooling(2)
    local gradOutput = torch.zeros(2, 2, 1, 1)
    gradOutput[1][1][1][1] = 2
    gradOutput[2][2][1][1] = 4

    gradInputTrue = torch.zeros(2, 2, 3, 3)
    gradInputTrue[1][1][1][1] = 1
    gradInputTrue[1][1][1][2] = 1
    gradInputTrue[2][2][3][3] = 2
    gradInputTrue[2][2][3][2] = 2
    output = m:forward(input)
    gradInput = m:backward(input, gradOutput)
    tester:eq(gradInput, gradInputTrue)

end

function mytest.testGlobalkMaxPoolingBackwardSingle()

    local input = torch.zeros(2, 3, 3)
    input[1][1][1] = 4
    input[1][1][2] = 2
    input[2][3][3] = 2
    input[2][3][2] = 1

    local m = nn.GlobalkMaxPooling()
    local gradOutput = torch.zeros(2, 1, 1)

    local output = m:forward(input)
    local gradInput = m:backward(input, gradOutput)
    tester:eq(gradInput, torch.zeros(2, 3, 3))

    gradOutput[1][1][1] = 2
    gradOutput[2][1][1] = 3
    gradInput = m:backward(input, gradOutput)

    gradInputTrue = torch.zeros(2, 3, 3)
    gradInputTrue[1][1][1] = 2
    gradInputTrue[2][3][3] = 3
    tester:eq(gradInput, gradInputTrue)

    local m = nn.GlobalkMaxPooling(2)
    local gradOutput = torch.zeros(2, 1, 1)
    gradOutput[1][1][1] = 2
    gradOutput[2][1][1] = 4

    gradInputTrue = torch.zeros(2, 3, 3)
    gradInputTrue[1][1][1] = 1
    gradInputTrue[1][1][2] = 1
    gradInputTrue[2][3][3] = 2
    gradInputTrue[2][3][2] = 2
    output = m:forward(input)
    gradInput = m:backward(input, gradOutput)
    tester:eq(gradInput, gradInputTrue)

end


-------------------------------------------------------------------------------
-- Test LogSumExpPooling
-------------------------------------------------------------------------------

function mytest.testLogSumExpPoolingForwardBatch()

    local input = torch.zeros(2, 2, 3, 3)
    input[1][1][1][1] = 4
    input[1][1][1][2] = 2
    input[2][2][3][3] = 2


    local beta = 1
    local outputTrue = torch.zeros(2, 2, 1, 1)
    outputTrue[1][1][1][1] = 1 / beta * math.log((math.exp(beta * 2) + math.exp(beta * 4) + 7) / 9)
    outputTrue[2][2][1][1] = 1 / beta * math.log((math.exp(beta * 2) + 8) / 9)

    local m = nn.LogSumExpPooling()
    local output = m:forward(input)
    tester:eq(output, outputTrue, 1e-10)

    beta = 10
    outputTrue = torch.zeros(2, 2, 1, 1)
    outputTrue[1][1][1][1] = 1 / beta * math.log((math.exp(beta * 2) + math.exp(beta * 4) + 7) / 9)
    outputTrue[2][2][1][1] = 1 / beta * math.log((math.exp(beta * 2) + 8) / 9)

    m = nn.LogSumExpPooling(beta)
    output = m:forward(input)
    tester:eq(output, outputTrue, 1e-10)

    beta = 100
    outputTrue = torch.zeros(2, 2, 1, 1)
    outputTrue[1][1][1][1] = 1 / beta * math.log((math.exp(beta * 2) + math.exp(beta * 4) + 7) / 9)
    outputTrue[2][2][1][1] = 1 / beta * math.log((math.exp(beta * 2) + 8) / 9)

    m = nn.LogSumExpPooling(beta)
    output = m:forward(input)
    tester:eq(output, outputTrue, 1e-10)

    beta = 0.1
    outputTrue = torch.zeros(2, 2, 1, 1)
    outputTrue[1][1][1][1] = 1 / beta * math.log((math.exp(beta * 2) + math.exp(beta * 4) + 7) / 9)
    outputTrue[2][2][1][1] = 1 / beta * math.log((math.exp(beta * 2) + 8) / 9)

    m = nn.LogSumExpPooling(beta)
    output = m:forward(input)
    tester:eq(output, outputTrue, 1e-10)

    beta = 0.01
    outputTrue = torch.zeros(2, 2, 1, 1)
    outputTrue[1][1][1][1] = 1 / beta * math.log((math.exp(beta * 2) + math.exp(beta * 4) + 7) / 9)
    outputTrue[2][2][1][1] = 1 / beta * math.log((math.exp(beta * 2) + 8) / 9)

    m = nn.LogSumExpPooling(beta)
    output = m:forward(input)
    tester:eq(output, outputTrue, 1e-10)

    beta = 0.001
    outputTrue = torch.zeros(2, 2, 1, 1)
    outputTrue[1][1][1][1] = 1 / beta * math.log((math.exp(beta * 2) + math.exp(beta * 4) + 7) / 9)
    outputTrue[2][2][1][1] = 1 / beta * math.log((math.exp(beta * 2) + 8) / 9)

    m = nn.LogSumExpPooling(beta)
    output = m:forward(input)
    tester:eq(output, outputTrue, 1e-10)

    beta = 1e-4
    outputTrue = torch.zeros(2, 2, 1, 1)
    outputTrue[1][1][1][1] = 1 / beta * math.log((math.exp(beta * 2) + math.exp(beta * 4) + 7) / 9)
    outputTrue[2][2][1][1] = 1 / beta * math.log((math.exp(beta * 2) + 8) / 9)

    m = nn.LogSumExpPooling(beta)
    output = m:forward(input)
    tester:eq(output, outputTrue, 1e-10)

end

function mytest.testLogSumExpPoolingForwardSingle()

    local input = torch.zeros(2, 3, 3)
    input[1][1][1] = 4
    input[1][1][2] = 2
    input[2][3][3] = 2


    local beta = 1
    local outputTrue = torch.zeros(2, 1, 1)
    outputTrue[1][1][1] = 1 / beta * math.log((math.exp(beta * 2) + math.exp(beta * 4) + 7) / 9)
    outputTrue[2][1][1] = 1 / beta * math.log((math.exp(beta * 2) + 8) / 9)

    local m = nn.LogSumExpPooling()
    local output = m:forward(input)
    tester:eq(output, outputTrue, 1e-10)

    beta = 100
    outputTrue = torch.zeros(2, 1, 1)
    outputTrue[1][1][1] = 1 / beta * math.log((math.exp(beta * 2) + math.exp(beta * 4) + 7) / 9)
    outputTrue[2][1][1] = 1 / beta * math.log((math.exp(beta * 2) + 8) / 9)

    m = nn.LogSumExpPooling(beta)
    output = m:forward(input)
    tester:eq(output, outputTrue, 1e-10)

    beta = 1e-4
    outputTrue = torch.zeros(2, 1, 1)
    outputTrue[1][1][1] = 1 / beta * math.log((math.exp(beta * 2) + math.exp(beta * 4) + 7) / 9)
    outputTrue[2][1][1] = 1 / beta * math.log((math.exp(beta * 2) + 8) / 9)

    m = nn.LogSumExpPooling(beta)
    output = m:forward(input)
    tester:eq(output, outputTrue, 1e-10)

end

function mytest.testLogSumExpPoolingBackwardBatch()

    local input = torch.zeros(2, 2, 3, 3)
    input[1][1][1][1] = 4
    input[1][1][1][2] = 2
    input[2][2][3][3] = 2

    local m = nn.LogSumExpPooling()
    local gradOutput = torch.zeros(2, 2, 1, 1)
    local output = m:forward(input)
    local gradInput = m:backward(input, gradOutput)
    tester:eq(gradInput, torch.zeros(2, 2, 3, 3))


    input = torch.ones(2, 2, 3, 3)
    gradOutput = torch.ones(2, 2, 1, 1)
    gradInput = m:backward(input, gradOutput)
    local beta = 1
    local value = math.exp(beta * 1) / (9 * math.exp(beta * 1))
    gradInputTrue = torch.ones(2, 2, 3, 3):mul(value)
    tester:eq(gradInput, gradInputTrue, 1e-10)


    beta = 1
    m = nn.LogSumExpPooling(beta)
    input = torch.ones(2, 2, 3, 3)
    input[1][1][1][1] = 2
    gradOutput = torch.ones(2, 2, 1, 1)
    gradInput = m:backward(input, gradOutput)
    local value1 = math.exp(beta * 1) / (8 * math.exp(beta * 1) + math.exp(beta * 2))
    local value2 = math.exp(beta * 1) / (9 * math.exp(beta * 1))
    gradInputTrue = torch.ones(2, 2, 3, 3)
    gradInputTrue[1][1]:mul(value1)
    gradInputTrue[1][2]:mul(value2)
    gradInputTrue[2]:mul(value2)
    gradInputTrue[1][1][1][1] = math.exp(beta * 2) / (8 * math.exp(beta * 1) + math.exp(beta * 2))
    tester:eq(gradInput, gradInputTrue, 1e-10)

    beta = 10
    m = nn.LogSumExpPooling(beta)
    input = torch.ones(2, 2, 3, 3)
    input[1][1][1][1] = 2
    gradOutput = torch.ones(2, 2, 1, 1)
    gradInput = m:backward(input, gradOutput)
    value1 = math.exp(beta * 1) / (8 * math.exp(beta * 1) + math.exp(beta * 2))
    value2 = math.exp(beta * 1) / (9 * math.exp(beta * 1))
    gradInputTrue = torch.ones(2, 2, 3, 3)
    gradInputTrue[1][1]:mul(value1)
    gradInputTrue[1][2]:mul(value2)
    gradInputTrue[2]:mul(value2)
    gradInputTrue[1][1][1][1] = math.exp(beta * 2) / (8 * math.exp(beta * 1) + math.exp(beta * 2))
    tester:eq(gradInput, gradInputTrue, 1e-10)

    beta = 100
    m = nn.LogSumExpPooling(beta)
    input = torch.ones(2, 2, 3, 3)
    input[1][1][1][1] = 2
    gradOutput = torch.ones(2, 2, 1, 1)
    gradInput = m:backward(input, gradOutput)
    value1 = math.exp(beta * 1) / (8 * math.exp(beta * 1) + math.exp(beta * 2))
    value2 = math.exp(beta * 1) / (9 * math.exp(beta * 1))
    gradInputTrue = torch.ones(2, 2, 3, 3)
    gradInputTrue[1][1]:mul(value1)
    gradInputTrue[1][2]:mul(value2)
    gradInputTrue[2]:mul(value2)
    gradInputTrue[1][1][1][1] = math.exp(beta * 2) / (8 * math.exp(beta * 1) + math.exp(beta * 2))
    tester:eq(gradInput, gradInputTrue, 1e-10)

    beta = 0.1
    m = nn.LogSumExpPooling(beta)
    input = torch.ones(2, 2, 3, 3)
    input[1][1][1][1] = 2
    gradOutput = torch.ones(2, 2, 1, 1)
    gradInput = m:backward(input, gradOutput)
    value1 = math.exp(beta * 1) / (8 * math.exp(beta * 1) + math.exp(beta * 2))
    value2 = math.exp(beta * 1) / (9 * math.exp(beta * 1))
    gradInputTrue = torch.ones(2, 2, 3, 3)
    gradInputTrue[1][1]:mul(value1)
    gradInputTrue[1][2]:mul(value2)
    gradInputTrue[2]:mul(value2)
    gradInputTrue[1][1][1][1] = math.exp(beta * 2) / (8 * math.exp(beta * 1) + math.exp(beta * 2))
    tester:eq(gradInput, gradInputTrue, 1e-10)

    beta = 0.01
    m = nn.LogSumExpPooling(beta)
    input = torch.ones(2, 2, 3, 3)
    input[1][1][1][1] = 2
    gradOutput = torch.ones(2, 2, 1, 1)
    gradInput = m:backward(input, gradOutput)
    value1 = math.exp(beta * 1) / (8 * math.exp(beta * 1) + math.exp(beta * 2))
    value2 = math.exp(beta * 1) / (9 * math.exp(beta * 1))
    gradInputTrue = torch.ones(2, 2, 3, 3)
    gradInputTrue[1][1]:mul(value1)
    gradInputTrue[1][2]:mul(value2)
    gradInputTrue[2]:mul(value2)
    gradInputTrue[1][1][1][1] = math.exp(beta * 2) / (8 * math.exp(beta * 1) + math.exp(beta * 2))
    tester:eq(gradInput, gradInputTrue, 1e-10)


    beta = 0.001
    m = nn.LogSumExpPooling(beta)
    input = torch.ones(2, 2, 3, 3)
    input[1][1][1][1] = 2
    gradOutput = torch.ones(2, 2, 1, 1)
    gradInput = m:backward(input, gradOutput)
    value1 = math.exp(beta * 1) / (8 * math.exp(beta * 1) + math.exp(beta * 2))
    value2 = math.exp(beta * 1) / (9 * math.exp(beta * 1))
    gradInputTrue = torch.ones(2, 2, 3, 3)
    gradInputTrue[1][1]:mul(value1)
    gradInputTrue[1][2]:mul(value2)
    gradInputTrue[2]:mul(value2)
    gradInputTrue[1][1][1][1] = math.exp(beta * 2) / (8 * math.exp(beta * 1) + math.exp(beta * 2))
    tester:eq(gradInput, gradInputTrue, 1e-10)

end

function mytest.testLogSumExpPoolingBackwardSingle()

    local input = torch.zeros(2, 3, 3)
    input[1][1][1] = 4
    input[1][1][2] = 2
    input[2][3][3] = 2

    local m = nn.LogSumExpPooling()
    local gradOutput = torch.zeros(2, 1, 1)
    local output = m:forward(input)
    local gradInput = m:backward(input, gradOutput)
    tester:eq(gradInput, torch.zeros(2, 3, 3))


    local beta = 1
    input = torch.ones(2, 3, 3)
    gradOutput = torch.ones(2, 1, 1)
    gradInput = m:backward(input, gradOutput)
    local value = math.exp(beta * 1) / (9 * math.exp(beta * 1))
    gradInputTrue = torch.ones(2, 3, 3):mul(value)
    tester:eq(gradInput, gradInputTrue, 1e-10)

    m = nn.LogSumExpPooling(beta)
    input = torch.ones(2, 3, 3)
    input[1][1][1] = 2
    gradOutput = torch.ones(2, 1, 1)
    gradInput = m:backward(input, gradOutput)
    local value1 = math.exp(beta * 1) / (8 * math.exp(beta * 1) + math.exp(beta * 2))
    local value2 = math.exp(beta * 1) / (9 * math.exp(beta * 1))
    gradInputTrue = torch.ones(2, 3, 3)
    gradInputTrue[1]:mul(value1)
    gradInputTrue[2]:mul(value2)
    gradInputTrue[1][1][1] = math.exp(beta * 2) / (8 * math.exp(beta * 1) + math.exp(beta * 2))
    tester:eq(gradInput, gradInputTrue, 1e-10)

    beta = 100
    m = nn.LogSumExpPooling(beta)
    input = torch.ones(2, 3, 3)
    input[1][1][1] = 2
    gradOutput = torch.ones(2, 1, 1)
    gradInput = m:backward(input, gradOutput)
    local value1 = math.exp(beta * 1) / (8 * math.exp(beta * 1) + math.exp(beta * 2))
    local value2 = math.exp(beta * 1) / (9 * math.exp(beta * 1))
    gradInputTrue = torch.ones(2, 3, 3)
    gradInputTrue[1]:mul(value1)
    gradInputTrue[2]:mul(value2)
    gradInputTrue[1][1][1] = math.exp(beta * 2) / (8 * math.exp(beta * 1) + math.exp(beta * 2))
    tester:eq(gradInput, gradInputTrue, 1e-10)

    beta = 0.001
    m = nn.LogSumExpPooling(beta)
    input = torch.ones(2, 3, 3)
    input[1][1][1] = 2
    gradOutput = torch.ones(2, 1, 1)
    gradInput = m:backward(input, gradOutput)
    local value1 = math.exp(beta * 1) / (8 * math.exp(beta * 1) + math.exp(beta * 2))
    local value2 = math.exp(beta * 1) / (9 * math.exp(beta * 1))
    gradInputTrue = torch.ones(2, 3, 3)
    gradInputTrue[1]:mul(value1)
    gradInputTrue[2]:mul(value2)
    gradInputTrue[1][1][1] = math.exp(beta * 2) / (8 * math.exp(beta * 1) + math.exp(beta * 2))
    tester:eq(gradInput, gradInputTrue, 1e-10)

end


tester:add(mytest)
tester:run()
