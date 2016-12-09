-- compatible CUDA
require 'torch'
require 'nn'

local GlobalAveragePooling, parent = torch.class('nn.GlobalAveragePooling', 'nn.Module')

function GlobalAveragePooling:__init()
    parent.__init(self)
    self.output = torch.Tensor()
end

function GlobalAveragePooling:updateOutput(input)

    local batchSize = 0
    local numChannels = 0
    local h = 0
    local w = 0

    if input:dim() == 4 then -- batch
        batchSize = input:size(1)
        numChannels = input:size(2)
        h = input:size(3)
        w = input:size(4)
    elseif input:dim() == 3 then -- image
        batchSize = 1
        numChannels = input:size(1)
        h = input:size(2)
        w = input:size(3)
    else
        print('error in GlobalAveragePooling:updateOutput - incorrect input size')
    end

    self.output:typeAs(input):resize(batchSize, numChannels, 1, 1)
    local x = input:view(batchSize, numChannels, h*w)

    -- Compute the max
    local sum = torch.sum(x, 3)
    self.output = sum:div(h*w)

    if input:dim() == 4 then
        self.output = self.output:view(batchSize, numChannels, 1, 1)
    elseif input:dim() == 3 then
        self.output = self.output:view(numChannels, 1, 1)
    end

    return self.output
end

function GlobalAveragePooling:updateGradInput(input, gradOutput)

    if input:dim() == 4 then -- batch
        batchSize = input:size(1)
        numChannels = input:size(2)
        h = input:size(3)
        w = input:size(4)
    elseif input:dim() == 3 then -- image
        batchSize = 1
        numChannels = input:size(1)
        h = input:size(2)
        w = input:size(3)
    else
        print('error in GlobalAveragePooling:updateGradInput - incorrect input size')
    end

    local y = gradOutput:clone():view(batchSize, numChannels, 1)
    self.gradInput = torch.repeatTensor(y, 1, 1, h*w)
    self.gradInput:div(h*w)

    if input:dim() == 4 then
        self.gradInput = self.gradInput:view(batchSize, numChannels, h, w)
    elseif input:dim() == 3 then
        self.gradInput = self.gradInput:view(numChannels, h, w)
    end

    return self.gradInput
end

function GlobalAveragePooling:empty()
    self.gradInput:resize()
    self.gradInput:storage():resize(0)
    self.output:resize()
    self.output:storage():resize(0)
end

function GlobalAveragePooling:__tostring__()
    local s =  string.format('%s()', torch.type(self))
    return s
end
