-- compatible CUDA
require 'torch'
require 'nn'

local GlobalMaxPooling, parent = torch.class('nn.GlobalMaxPooling', 'nn.Module')

function GlobalMaxPooling:__init()
    parent.__init(self)

    self.output = torch.Tensor()
    self.indices = torch.Tensor()
end

function GlobalMaxPooling:updateOutput(input)

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
        print('error in GlobalMaxPooling:updateOutput - incorrect input size')
    end

    self.output:typeAs(input):resize(batchSize, numChannels, 1, 1)
    local x = input:view(batchSize, numChannels, h*w)

    -- Compute the max
    local max, indices = torch.max(x, 3)
    self.indices = indices
    self.output = max

    if input:dim() == 4 then
        self.output = self.output:view(batchSize, numChannels, 1, 1)
    elseif input:dim() == 3 then
        self.output = self.output:view(numChannels, 1, 1)
    end

    return self.output
end

function GlobalMaxPooling:updateGradInput(input, gradOutput)

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
        print('error in GlobalMaxPooling:updateGradInput - incorrect input size')
    end

    local y = gradOutput:clone():view(batchSize, numChannels, 1)
    self.gradInput = torch.zeros(batchSize, numChannels, h*w):typeAs(input)
    self.gradInput:scatter(3, self.indices, y)

    if input:dim() == 4 then
        self.gradInput = self.gradInput:view(batchSize, numChannels, h, w)
    elseif input:dim() == 3 then
        self.gradInput = self.gradInput:view(numChannels, h, w)
    end

    return self.gradInput
end

function GlobalMaxPooling:empty()
    self.gradInput:resize()
    self.gradInput:storage():resize(0)
    self.output:resize()
    self.output:storage():resize(0)
    self.indices:resize()
    self.indices:storage():resize(0)
end

function GlobalMaxPooling:__tostring__()
    local s =  string.format('%s()', torch.type(self))
    return s
end
